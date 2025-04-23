import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random

from vllm import LLM, SamplingParams
from utils.helpers import get_tokenizer, make_dataset
from utils.post_processor import PostProcessor
from torch.utils.data import Subset

class Inference:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = get_tokenizer(cfg)
        self.tokenizer.save_pretrained("./tokenizer")
        self.llm = self._setup_llm()
        self.sampling_params = self._get_sampling_params()

    def _setup_llm(self):
        return LLM(
            model=self.cfg.test.model_path,
            tokenizer="./tokenizer",
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
            gpu_memory_utilization=self.cfg.get("gpu_memory_util", None),
        )

    def _get_sampling_params(self):
        stop_tokens = (
            ['|endofevent|'] if self.cfg.data.add_end_of_event
            else [f'[DAY_{i}]' for i in range(1, self.cfg.data.max_day_len + 1)]
        )
        return SamplingParams(
            temperature=0,
            top_k=1,
            stop=stop_tokens,
            skip_special_tokens=False,
            logprobs=1 if self.cfg.test.logprobs else None
        )

    def decode_output(self, output):
        gen_text = ' '.join([o.text for o in output.outputs])
        log_probs = output.outputs[0].logprobs if output.outputs else []
        return gen_text, log_probs

    def run_inference(self, dataset):
        results = []
        for prompt_token, label_token, prev_token, mean_prev in tqdm(dataset):
            output = self.llm.generate(prompt_token_ids=[prompt_token.tolist()],
                                       sampling_params=self.sampling_params)[0]
            gen_text, log_probs = self.decode_output(output)

            result = {
                "prompt": self.tokenizer.decode(output.prompt_token_ids, add_special_tokens=False),
                "generated_text": gen_text,
                "label_text": (
                    label_token.item()
                    if self.cfg.data.num_bucket else self.tokenizer.decode(label_token, add_special_tokens=False)
                ),
                "previous_text": self.tokenizer.decode(prev_token, add_special_tokens=False),
            }
            if self.cfg.test.logprobs:
                result["log_probs"] = log_probs
            else:
                result["mean_previous_value"] = mean_prev

            results.append(result)
        return results

    def run_inference_num_bucket(self, dataset):
        num_tokens = self.tokenizer.encode(
            " ".join([f"[NUM_{i}]" for i in range(1, self.cfg.data.num_bucket_num + 1)]),
            add_special_tokens=False
        )
        results = []
        for prompt_token, label_token, prev_token, _ in tqdm(dataset):
            output = self.llm.generate(prompt_token_ids=[prompt_token.tolist()],
                                       sampling_params=self.sampling_params)[0]
            _, log_probs_raw = self.decode_output(output)

            prob_dict = {
                self.tokenizer.decode(k, add_special_tokens=False): np.exp(log_probs_raw[k].logprob)
                for k in num_tokens if k in log_probs_raw
            }
            results.append({
                "prompt": self.tokenizer.decode(output.prompt_token_ids, add_special_tokens=False),
                "generated_text": ' '.join([o.text for o in output.outputs]),
                "label_text": label_token.item(),
                "previous_text": self.tokenizer.decode(prev_token, add_special_tokens=False),
                "prob_dict": prob_dict,
            })
        return results

    def save_results(self, results, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(results, f)

    
    def predict(self):
        torch.cuda.empty_cache()
        _, _, test_prompt_dataset, _ = make_dataset(self.cfg, get_tokenizer(self.cfg), prompt_test=True)

        if self.cfg.mode.debugging_mode:
            test_prompt_dataset = Subset(test_prompt_dataset, random.sample(
                range(len(test_prompt_dataset)), self.cfg.test.sample_size
            ))

        results = (
            self.run_inference_num_bucket(test_prompt_dataset)
            if self.cfg.data.num_bucket else
            self.run_inference(test_prompt_dataset)
        )
        
        # Post-process
        post_processor = PostProcessor(self.cfg, self.tokenizer)
        save_path = post_processor.post_process(results, self.cfg.data_name + "_" + self.cfg.data_path.split('/')[-2])
        
        # save path
        return save_