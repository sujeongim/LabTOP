# Merged and simplified version of num_bucket.py and base.py

import os
import re
import json
import torch
import wandb
import logging
import multiprocessing
import numpy as np
from copy import deepcopy
from datetime import timedelta
from torch import nn
from tqdm import tqdm
import time
from torch.utils.data import Subset
from omegaconf import OmegaConf
from accelerate import Accelerator, InitProcessGroupKwargs
from core.utils.helpers import get_optimal_num_workers, EarlyStopping
#from inference import EvalValue
from transformers import get_scheduler
from core.data.dataloader import EHRGPTDataLoader, PromptTestDataLoader

logging.basicConfig(level=logging.WARNING)
logging.getLogger("accelerate").setLevel(logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Trainer:
    def __init__(self, cfg, model, tokenizer, train_dataset, valid_dataset, valid_prompt_dataset, optimizer, criterion, device):
        self.cfg, self.model, self.tokenizer, self.device = cfg, model, tokenizer, device
        self.valid_prompt_dataset = Subset(valid_prompt_dataset, range(100000)) if valid_prompt_dataset else None
       
        if cfg.debugging_mode:
            self.train_dataset = Subset(train_dataset, range(1000))
            self.valid_dataset = Subset(valid_dataset, range(100))
        else:
            self.train_dataset = train_dataset
            self.valid_dataset = Subset(valid_dataset, range(10000))
            
        self.optimizer, self.criterion = optimizer, criterion

        self._setup_accelerator()
        self._prepare_dataloaders()
        self._prepare_training()
        self._setup_eval_module()
        self._prepare_dirs()
        
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'valid_loss': [], 'valid_acc': [],
            'learning_rates': [], 'epoch_times': []
        }

    def _setup_accelerator(self):
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        self.accelerator = Accelerator(
            kwargs_handlers=[kwargs], 
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_steps,
            log_with="wandb" if (self.cfg.train.use_wandb) else None # and not self.cfg.debugging_mode
        )
        self.accelerator.init_trackers("labtop", config=OmegaConf.to_container(self.cfg, resolve=True))

    def _prepare_dataloaders(self):
        self.batch_size = self.cfg.train.batch_size // self.cfg.train.gradient_accumulation_steps
        # multiprocessing 오류 방지
        try:
            num_workers = get_optimal_num_workers(self.batch_size)
        except (NameError, AttributeError) as e:
            if self.accelerator.is_main_process:
                self.accelerator.print(f"Warning: Could not determine optimal workers: {e}")
            num_workers = 4  
        
        self.train_dataloader = EHRGPTDataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=num_workers).dataloader
        self.valid_dataloader = EHRGPTDataLoader(self.valid_dataset, self.batch_size, shuffle=False, num_workers=num_workers).dataloader
        self.valid_prompt_dataloader = PromptTestDataLoader(self.valid_prompt_dataset, self.batch_size, shuffle=False, num_workers=num_workers).dataloader
        
    def _prepare_training(self):
        steps = len(self.train_dataloader) * self.cfg.train.epochs // self.cfg.train.gradient_accumulation_steps
        warmup_steps = int(0.1 * steps)
        self.scheduler = get_scheduler("linear", self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps)
        components = [self.model, self.optimizer, self.scheduler, self.train_dataloader, self.valid_dataloader]
        self.model, self.optimizer, self.scheduler, self.train_dataloader, self.valid_dataloader = self.accelerator.prepare(*components)
        if self.cfg.train.model_path:
            self.accelerator.load_state(self.cfg.train.model_path)

    def _setup_eval_module(self):
        self.eos_token_ids = [
            self.tokenizer.encode("|endofevent|" if self.cfg.data.add_end_of_event else f"[DAY_{d}]", add_special_tokens=False)[0]
            for d in (range(1, self.cfg.data.max_day_len + 1) if not self.cfg.data.add_end_of_event else [0])
        ]
        #self.eval_value = EvalValue(self.cfg, self.accelerator, self.tokenizer, self.eos_token_ids, len(self.valid_prompt_dataset))

    def _prepare_dirs(self):
        data_tag = '_'.join(self.cfg.data_path.split('/')[-2:]) if 'lab' not in self.cfg.data_path.split('/')[-1] else self.cfg.data_path.split('/')[-1]
        if self.cfg.train.train_only_value_valueuom:
            weekdays = ['[MON]', '[TUE]', '[WED]', '[THU]', '[FRI]', '[SAT]', '[SUN]'] if self.cfg.data.add_weekday else []
            self.week_time_tokens = weekdays + [f'[{h:02}h]' for h in range(24)] + [f'[{m:02}m]' for m in range(0, 60, 10)]
            self.week_time_token_ids = torch.tensor([self.tokenizer.encode(t, add_special_tokens=False)[0] for t in self.week_time_tokens])
            data_tag += "_train_only_value"
        if self.cfg.mode.debugging_mode: data_tag += "_debug"
        data_tag += f"_{self.cfg.max_seq_len}_seed{self.cfg.train.seed}"
        self.model_dir = f"./trained_models/{data_tag}_head{self.cfg.model.n_heads}_layer{self.cfg.model.n_layers}_dim{self.cfg.model.hidden_dim}"
        if not self.cfg.train.model_path and os.path.exists(self.model_dir) and self.accelerator.is_main_process:
            print("Remove existing model directory"); input()
            os.system(f"rm -rf {self.model_dir}")

    def _forward_pass(self, inputs):
        masks = (inputs != 0).float()
        return self.model(input_ids=inputs, attention_mask=masks)

    def _compute_loss_and_accuracy(self, logits, labels):
        labels, logits = labels[:, 1:], logits[:, :-1]
        if self.cfg.train.train_only_value_valueuom:
            mask = self.train_type[:, 1:] == 1
            labels, logits = labels[mask], logits[mask]
        if labels.numel() == 0:
            return None, None
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()
        return loss, accuracy

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_loss, total_acc = [], []
        
        # main process에서만 progress bar 표시
        if self.accelerator.is_main_process:
            valid_pbar = tqdm(self.valid_dataloader, desc="Validation", leave=False)
        else:
            valid_pbar = self.valid_dataloader
            
        for inputs, train_type, pos_enc in valid_pbar:
            inputs = inputs.to(self.accelerator.device)
            self.train_type = train_type.to(self.accelerator.device)
            logits = self._forward_pass(inputs)
            loss, acc = self._compute_loss_and_accuracy(logits, deepcopy(inputs))
            if loss is not None:
                total_loss.append(loss.item())
                total_acc.append(acc)
                
        if total_loss:
            avg_loss, avg_acc = map(np.mean, (total_loss, total_acc))
            if self.accelerator.is_main_process:
                self.accelerator.print(f"[Valid] Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
            return avg_loss, avg_acc
        return 0.0, 0.0

    def _log_metrics(self, epoch, train_loss, train_acc, valid_loss, valid_acc, lr, epoch_time):
        """메트릭 로깅 및 히스토리 저장"""
        # 히스토리 저장
        self.training_history['train_loss'].append(train_loss)
        self.training_history['train_acc'].append(train_acc)
        self.training_history['valid_loss'].append(valid_loss)
        self.training_history['valid_acc'].append(valid_acc)
        self.training_history['learning_rates'].append(lr)
        self.training_history['epoch_times'].append(epoch_time)
        
        # wandb 로깅 (main process에서만)
        if self.cfg.train.use_wandb and self.accelerator.is_main_process:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "valid/loss": valid_loss,
                "valid/accuracy": valid_acc,
                "train/learning_rate": lr,
                "train/epoch_time": epoch_time
            }
            self.accelerator.log(log_dict, step=epoch)

    def _print_training_summary(self):
        """학습 요약 정보 출력"""
        if not self.accelerator.is_main_process:
            return
            
        total_time = sum(self.training_history['epoch_times'])
        best_valid_acc = max(self.training_history['valid_acc'])
        best_epoch = self.training_history['valid_acc'].index(best_valid_acc) + 1
        
        self.accelerator.print("\n" + "="*50)
        self.accelerator.print("TRAINING SUMMARY")
        self.accelerator.print("="*50)
        self.accelerator.print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f}m)")
        self.accelerator.print(f"Average time per epoch: {np.mean(self.training_history['epoch_times']):.2f}s")
        self.accelerator.print(f"Best validation accuracy: {best_valid_acc:.4f} (Epoch {best_epoch})")
        self.accelerator.print(f"Final train loss: {self.training_history['train_loss'][-1]:.4f}")
        self.accelerator.print(f"Final valid loss: {self.training_history['valid_loss'][-1]:.4f}")
        self.accelerator.print("="*50)

    
    def train(self, save_path=None):
        early_stop = EarlyStopping(patience=self.cfg.train.patience, mode='min')
        best_loss = np.inf

        # wandb 초기화 (main process에서만)
        if self.cfg.train.use_wandb and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name="LabTOP",
                config=dict(self.cfg.train),
                init_kwargs={"wandb": {"name": f"exp_{self.cfg.train.seed}"}}
            )

        if self.accelerator.is_main_process:
            self.accelerator.print(f"Starting training for {self.cfg.train.epochs} epochs...")
            self.accelerator.print(f"Batch size: {self.cfg.train.batch_size}")
            self.accelerator.print(f"Gradient accumulation steps: {self.cfg.train.gradient_accumulation_steps}")
            self.accelerator.print(f"Effective batch size: {self.cfg.train.batch_size * self.cfg.train.gradient_accumulation_steps}")

        for epoch in range(self.cfg.train.epochs):
            epoch_start_time = time.time()
            self.model.train()
            total_loss, total_acc = [], []
            
            # main process에서만 progress bar 표시
            if self.accelerator.is_main_process:
                train_pbar = tqdm(
                    self.train_dataloader, 
                    desc=f"Epoch {epoch+1}/{self.cfg.train.epochs}",
                    leave=False
                )
            else:
                train_pbar = self.train_dataloader
                
            for step, (inputs, train_type, pos_enc) in enumerate(train_pbar):
                inputs = inputs.to(self.accelerator.device)
                self.train_type = train_type.to(self.accelerator.device)
                
                with self.accelerator.accumulate(self.model):
                    logits = self._forward_pass(inputs)
                    loss, acc = self._compute_loss_and_accuracy(logits, deepcopy(inputs))
                    if loss is None: 
                        continue
                    
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping
                    if self.accelerator.sync_gradients:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_norm_clip)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    total_loss.append(loss.item())
                    total_acc.append(acc)
                    
                    # progress bar 업데이트 (main process에서만)
                    if self.accelerator.is_main_process and isinstance(train_pbar, tqdm):
                        train_pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'acc': f'{acc:.4f}',
                            'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                        })
            
            
            # 에포크 결과 계산
            avg_loss, avg_acc = map(np.mean, (total_loss, total_acc))
            current_lr = self.scheduler.get_last_lr()[0]
            epoch_time = time.time() - epoch_start_time
            
            # Validation
            valid_loss, valid_acc = self._validate()

            # 메트릭 로깅
            self._log_metrics(epoch + 1, avg_loss, avg_acc, valid_loss, valid_acc, current_lr, epoch_time)

            # main process에서만 결과 출력
            if self.accelerator.is_main_process:
                self.accelerator.print(
                    f"[Epoch {epoch+1:3d}] "
                    f"Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f} | "
                    f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f} | "
                    f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
                )

            # Save best model (main process에서만)
            if valid_loss < best_loss and self.accelerator.is_main_process:
                best_loss = valid_loss
                self.accelerator.save_state(self.model_dir)
                self.accelerator.print(f"✓ New best model saved with loss: {best_loss:.4f}")

            # Early stopping 체크
            if early_stop.step(valid_loss):
                if self.accelerator.is_main_process:
                    self.accelerator.print("Early stopping triggered.")
                break

        # 학습 완료 후 요약 출력
        self._print_training_summary()

        # wandb 종료
        if self.cfg.train.use_wandb and self.accelerator.is_main_process:
            self.accelerator.end_training()

        if self.accelerator.is_main_process:
            self.accelerator.print(f"Training completed! Best model saved in: {self.model_dir}")