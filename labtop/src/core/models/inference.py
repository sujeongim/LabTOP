import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random

from core.utils.helpers import get_tokenizer, make_dataset
from core.utils.post_processor import PostProcessor
from torch.utils.data import Subset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

class Inference:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = get_tokenizer(cfg)
        self.model = self._load_local_model()
        
    def _load_local_model(self):
        """로컬 모델 로드"""
        print(f"Loading model from {self.cfg.test.model_path}")
        
        try:
            # Hugging Face 모델 로드
            model = AutoModelForCausalLM.from_pretrained(
                self.cfg.test.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.device_count() > 1 else None,
                trust_remote_code=True
            )
            
            if torch.cuda.device_count() <= 1:
                model = model.to(self.device)
                
        except Exception as e:
            print(f"Failed to load with AutoModelForCausalLM: {e}")
            # 커스텀 모델 로드 (LabTOPModel 등)
            from core.models.model import LabTOPModel
            model = LabTOPModel(self.cfg, self.tokenizer)
            
            # 체크포인트 로드
            if os.path.exists(os.path.join(self.cfg.test.model_path, "pytorch_model.bin")):
                state_dict = torch.load(
                    os.path.join(self.cfg.test.model_path, "pytorch_model.bin"),
                    map_location=self.device
                )
                model.load_state_dict(state_dict)
            else:
                print("Warning: No model weights found")
                
            model = model.to(self.device)
            
        model.eval()
        return model

    def _get_stop_token_ids(self):
        """중지 토큰 ID 가져오기"""
        if self.cfg.data.add_end_of_event:
            stop_tokens = ['|endofevent|']
        else:
            stop_tokens = [f'[DAY_{i}]' for i in range(1, self.cfg.data.max_day_len + 1)]
            
        stop_token_ids = []
        for token in stop_tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if token_ids:
                stop_token_ids.append(token_ids[0])
        return stop_token_ids

    @torch.no_grad()
    def generate_text(self, prompt_tokens, max_new_tokens=50, temperature=0.0):
        """로컬 텍스트 생성"""
        input_ids = torch.tensor([prompt_tokens], device=self.device)
        stop_token_ids = self._get_stop_token_ids()
        
        generated_tokens = []
        
        for _ in range(max_new_tokens):
            # 모델 예측
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
            # 다음 토큰 예측
            next_token_logits = logits[0, -1, :]
            
            if temperature == 0.0:
                # Greedy sampling
                next_token_id = torch.argmax(next_token_logits, dim=-1)
            else:
                # Temperature sampling
                next_token_logits = next_token_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
                
            next_token_id = next_token_id.item()
            
            # 중지 조건 확인
            if next_token_id in stop_token_ids or next_token_id == self.tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token_id)
            
            # 입력에 새 토큰 추가
            input_ids = torch.cat([
                input_ids, 
                torch.tensor([[next_token_id]], device=self.device)
            ], dim=1)
            
            # 메모리 절약을 위한 길이 제한
            if input_ids.size(1) > self.cfg.get('max_sequence_length', 2048):
                input_ids = input_ids[:, -self.cfg.get('max_sequence_length', 2048):]
                
        return generated_tokens

    @torch.no_grad() 
    def get_token_probabilities(self, prompt_tokens, target_token_ids):
        """특정 토큰들의 확률 계산"""
        input_ids = torch.tensor([prompt_tokens], device=self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        
        prob_dict = {}
        for token_id in target_token_ids:
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            prob_dict[token_text] = probs[token_id].item()
            
        return prob_dict

    def run_inference(self, dataset):
        """일반 추론 실행"""
        results = []
        
        for prompt_token, label_token, prev_token, mean_prev in tqdm(dataset, desc="Running inference"):
            # 텍스트 생성
            generated_tokens = self.generate_text(
                prompt_token.tolist(), 
                max_new_tokens=self.cfg.test.get('max_new_tokens', 50),
                temperature=self.cfg.test.get('temperature', 0.0)
            )
            
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
            
            result = {
                "prompt": self.tokenizer.decode(prompt_token, add_special_tokens=False),
                "generated_text": generated_text,
                "label_text": (
                    label_token.item()
                    if self.cfg.data.num_bucket else 
                    self.tokenizer.decode(label_token, add_special_tokens=False)
                ),
                "previous_text": self.tokenizer.decode(prev_token, add_special_tokens=False),
                "mean_previous_value": mean_prev
            }
            
            results.append(result)
            
            # 메모리 정리
            if len(results) % 100 == 0:
                torch.cuda.empty_cache()
                
        return results

    def run_inference_num_bucket(self, dataset):
        """수치 버킷 추론 실행"""
        # 수치 토큰 ID들 가져오기
        num_tokens = self.tokenizer.encode(
            " ".join([f"[NUM_{i}]" for i in range(1, self.cfg.data.num_bucket_num + 1)]),
            add_special_tokens=False
        )
        
        results = []
        
        for prompt_token, label_token, prev_token, _ in tqdm(dataset, desc="Running num bucket inference"):
            # 확률 계산
            prob_dict = self.get_token_probabilities(prompt_token.tolist(), num_tokens)
            
            # 최고 확률 토큰으로 생성
            best_token_id = max(num_tokens, key=lambda x: prob_dict.get(
                self.tokenizer.decode([x], skip_special_tokens=False), 0
            ))
            generated_text = self.tokenizer.decode([best_token_id], skip_special_tokens=False)
            
            result = {
                "prompt": self.tokenizer.decode(prompt_token, add_special_tokens=False),
                "generated_text": generated_text,
                "label_text": label_token.item(),
                "previous_text": self.tokenizer.decode(prev_token, add_special_tokens=False),
                "prob_dict": prob_dict,
            }
            
            results.append(result)
            
            # 메모리 정리
            if len(results) % 100 == 0:
                torch.cuda.empty_cache()
                
        return results

    def save_results(self, results, save_path):
        """결과 저장"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {save_path}")

    def predict(self):
        """예측 실행"""
        print("Starting local inference...")
        torch.cuda.empty_cache()
        
        # 데이터셋 로드
        _, _, test_prompt_dataset, _ = make_dataset(self.cfg, get_tokenizer(self.cfg), prompt_test=True)

        if self.cfg.mode.debugging_mode:
            test_prompt_dataset = Subset(test_prompt_dataset, random.sample(
                range(len(test_prompt_dataset)), self.cfg.test.sample_size
            ))
            print(f"Using {len(test_prompt_dataset)} samples for debugging")

        # 추론 실행
        if self.cfg.data.num_bucket:
            results = self.run_inference_num_bucket(test_prompt_dataset)
        else:
            results = self.run_inference(test_prompt_dataset)
        
        print(f"Generated {len(results)} results")
        
        # 후처리
        post_processor = PostProcessor(self.cfg, self.tokenizer)
        save_path = post_processor.post_process(
            results, 
            self.cfg.data_name + "_" + self.cfg.data_path.split('/')[-2]
        )
        
        return save_path