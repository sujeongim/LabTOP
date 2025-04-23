# Merged and simplified version of num_bucket.py and base.py

import os
import re
import json
import torch
import wandb
import logging
import numpy as np
from copy import deepcopy
from datetime import timedelta
from torch import nn
from torch.utils.data import Subset
from omegaconf import OmegaConf
from transformers import AdamW, get_scheduler
from accelerate import Accelerator, InitProcessGroupKwargs
from core.utils.helpers import get_optimal_num_workers, EarlyStopping
#from inference import EvalValue
from core.data.dataloader import EHRGPTDataLoader, PromptTestDataLoader

logging.basicConfig(level=logging.WARNING)
logging.getLogger("accelerate").setLevel(logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Trainer:
    def __init__(self, cfg, model, tokenizer, train_dataset, valid_dataset, valid_prompt_dataset, device):
        self.cfg, self.model, self.tokenizer, self.device = cfg, model, tokenizer, device
        self.valid_prompt_dataset = Subset(valid_prompt_dataset, range(100000)) if not cfg.mode.debugging_mode else valid_prompt_dataset

        self._setup_accelerator()
        self._prepare_dataloaders(train_dataset, valid_dataset)
        self._prepare_training()
        self._setup_eval_module()
        self._prepare_dirs()

    def _setup_accelerator(self):
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        self.accelerator = Accelerator(kwargs_handlers=[kwargs], gradient_accumulation_steps=self.cfg.train.gradient_accumulation_steps)

    def _prepare_dataloaders(self, train_dataset, valid_dataset):
        self.batch_size = self.cfg.train.batch_size // self.cfg.train.gradient_accumulation_steps
        num_workers = get_optimal_num_workers(self.batch_size)
        self.train_dataloader = EHRGPTDataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=num_workers).dataloader
        self.valid_dataloader = EHRGPTDataLoader(valid_dataset, self.batch_size, shuffle=False, num_workers=num_workers).dataloader
        self.valid_prompt_dataloader = PromptTestDataLoader(self.valid_prompt_dataset, self.batch_size, shuffle=False, num_workers=num_workers).dataloader

    def _prepare_training(self):
        steps = len(self.train_dataloader) * self.cfg.train.epochs // self.cfg.train.gradient_accumulation_steps
        warmup_steps = int(0.1 * steps)
        self.optimizer = AdamW(self.model.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        self.scheduler = get_scheduler("linear", self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps)
        self.criterion = nn.CrossEntropyLoss()
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
        for inputs, train_type, pos_enc in self.valid_dataloader:
            inputs = inputs.to(self.accelerator.device)
            self.train_type = train_type.to(self.accelerator.device)
            logits = self._forward_pass(inputs)
            loss, acc = self._compute_loss_and_accuracy(logits, deepcopy(inputs), train_type)
            if loss is not None:
                total_loss.append(loss.item())
                total_acc.append(acc)
        if total_loss:
            avg_loss, avg_acc = map(np.mean, (total_loss, total_acc))
            self.accelerator.print(f"[Valid] Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
            return avg_loss, avg_acc
        return 0.0, 0.0


    def train(self):
        early_stop = EarlyStopping(patience=30, mode='min')
        best_loss = np.inf

        for epoch in range(self.cfg.train.epochs):
            self.model.train()
            total_loss, total_acc = [], []
            for inputs, train_type, pos_enc in self.train_dataloader:
                inputs = inputs.to(self.accelerator.device)
                self.train_type = train_type.to(self.accelerator.device)
                with self.accelerator.accumulate(self.model):
                    logits = self._forward_pass(inputs)
                    loss, acc = self._compute_loss_and_accuracy(logits, deepcopy(inputs))
                    if loss is None: continue
                    self.accelerator.backward(loss)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    total_loss.append(loss.item())
                    total_acc.append(acc)
            avg_loss, avg_acc = map(np.mean, (total_loss, total_acc))
            self.accelerator.print(f"[Train] Epoch {epoch+1} | Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

            # Validation
            valid_loss, valid_acc = self._validate()

            # Save best model
            if valid_loss < best_loss and self.accelerator.is_main_process:
                best_loss = valid_loss
                self.accelerator.save_state(self.model_dir)
                self.accelerator.print(f"New best model saved with acc: {best_loss:.4f}")

            if early_stop.step(valid_loss):
                self.accelerator.print("Early stopping triggered.")
                break
