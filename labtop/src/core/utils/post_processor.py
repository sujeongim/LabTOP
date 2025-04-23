import os
import re
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

class PostProcessor:
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.item_num_dict = {}
        self.item_text_dict = {}

        # Separator regex pattern based on timeoffset setting
        self.sep_info = self._init_separator()

        # Load representative bucket values if using num_bucket
        if cfg.data.num_bucket:
            self.representative_values = self._load_representative_values()

        # Load unit-of-measure set if available
        self.valueuom_set = self._load_uom_set()

    def _init_separator(self):
        if self.cfg.data.timeoffset == 'gap':
            return ']'
        if self.cfg.data.add_weekday:
            return r'\[DAY_\d+\] \[[A-Z]{3}\] \[\d{2}h\] \[\d{2}m\]'
        return r'\[DAY_\d+\] \[\d{2}h\] \[\d{2}m\]'

    def _load_representative_values(self):
        with open(f"{self.cfg.data_path}/representative_values_{self.cfg.data.num_bucket_num}", "rb") as f:
            raw = pickle.load(f)
        return {(k[0], k[1].replace(" ", "")): v for k, v in raw.items()}

    def _load_uom_set(self):
        path = f"/nfs_edlab/sjim/preprocessed_data/{self.cfg.data_name}/valueuom.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                return list(pickle.load(f))
        return None

    def _parse_numeric(self, text):
        """Extract float value from generated/label text, stripping units if present."""
        text = text.replace('[PAD]', '').replace(' ', '')
        for uom in self.valueuom_set or []:
            key = uom.lower().replace(' ', '')
            if key in text:
                text = text.replace(key, '')
                break
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if len(nums) == 2 and '-' in text:
            return (float(nums[0]) + float(nums[1])) / 2, True
        elif len(nums) == 1:
            return float(nums[0]), True
        return text, False

    def _estimate_bucket(self, item_key, prob_dict):
        """Estimate mean value using softmax-weighted representative values."""
        name = item_key.split('labevents')[-1].strip()
        means = self.representative_values.get(('labevents', name.replace(' ', '')), {})
        score = sum(prob_dict.get(f'[NUM_{n}]', 0) * means.get(n, 0)
                    for n in range(1, self.cfg.data.num_bucket_num + 1))
        return score, True

    def _record(self, prompt, result, label, prev, prob_dict):
        """Add one record to appropriate container (numerical or text-based)."""
        item_key = re.split(self.sep_info, prompt)[-1].strip()
        result, prev = result.replace(' ', ''), prev.replace(' ', '')
        label = label.item() if isinstance(label, torch.Tensor) else label.replace('[PAD]', '').replace(' ', '')

        # Filtering conditions
        if self.cfg.data.only_top_10 and item_key.split('labevents')[-1].strip() not in self.cfg.data.item_list:
            return
        if self.cfg.test.exclude_first_event and prev == '':
            return
        if self.cfg.test.only_first_event and prev != '':
            return

        # Parse and classify
        if self.cfg.data.num_bucket:
            pred, pred_ok = self._estimate_bucket(item_key, prob_dict)
            true, true_ok = label, True
            prev_val, _ = self._parse_numeric(prev)
        else:
            pred, pred_ok = self._parse_numeric(result)
            true, true_ok = self._parse_numeric(label)
            prev_val, _ = self._parse_numeric(prev)

        # Store
        target = self.item_num_dict if true_ok else self.item_text_dict
        if item_key not in target:
            target[item_key] = {
                "pred_values": [], "true_values": [],
                "prev_values": [], "mean_prev_values": []
            }
        target[item_key]["pred_values"].append(pred)
        target[item_key]["true_values"].append(true)

    def post_process(self, results, data_path):
        """Aggregate results per item and save as pickle."""
        for r in tqdm(results):
            self._record(
                r['prompt'],
                r['generated_text'],
                r['label_text'],
                r['previous_text'],
                r.get('prob_dict')
            )

        pred_true_dict = {
            k: pd.DataFrame({"true": v['true_values'], "pred": v['pred_values']})
            for k, v in self.item_num_dict.items()
        }

        model_dir = self.cfg.test.model_path.split("/")[-2]
        if self.cfg.mode.debugging_mode:
            model_dir += "_debug"

        os.makedirs("src/inference/inference_results/pkl_files", exist_ok=True)
        with open(f"src/inference/inference_results/pkl_files/{model_dir}.pkl", "wb") as f:
            pickle.dump(pred_true_dict, f)
        with open(f"src/inference/inference_results/pkl_files/{model_dir}_results.pkl", "wb") as f:
            pickle.dump(results, f)

        print(f"✅ Results saved to: src/inference/inference_results/pkl_files/{model_dir}.pkl")
        print(f"✅ Results saved to: src/inference/inference_results/pkl_files/{model_dir}_results.pkl")
        
        # return save_path
        return f"src/inference/inference_results/pkl_files/{model_dir}.pkl"