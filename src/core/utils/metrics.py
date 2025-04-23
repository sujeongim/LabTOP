import numpy as np
import pandas as pd
import pickle

class PerformanceEvaluator:
    def __init__(self, min_t=1, max_t=99, ref_lower=None, ref_upper=None):
        self.min_t = min_t
        self.max_t = max_t
        self.ref_lower = ref_lower
        self.ref_upper = ref_upper

    @staticmethod
    def task_success_rate(predictions, ground_truth):
        total = len(predictions)
        correct = sum(1 for pred, gt in zip(predictions, ground_truth) if pred == gt)
        return correct / total if total > 0 else 0.0

    @staticmethod
    def smape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        smape_values = np.zeros_like(y_true, dtype=np.float64)
        valid_mask = denominator != 0
        smape_values[valid_mask] = 100 * np.abs(y_true[valid_mask] - y_pred[valid_mask]) / denominator[valid_mask]
        smape_values[~valid_mask] = 0.0
        return np.mean(smape_values)

    @staticmethod
    def normalized_mae(true, pred, min_v, max_v):
        mae = np.mean(np.abs(true - pred))
        return mae / (max_v - min_v), mae
    
    @staticmethod
    def calculate_metrics(self, true, pred):
        new_true, new_pred = [], []
        for t, p in zip(true, pred):
            if isinstance(t, str) or isinstance(p, str):
                continue
            new_true.append(t)
            new_pred.append(p)

        true = np.array(new_true)
        pred = np.array(new_pred)

        min_v, max_v = np.percentile(true, [self.min_t, self.max_t])
        mask = (true >= min_v) & (true <= max_v) & (true == true)

        if self.ref_lower is not None and self.ref_upper is not None:
            ref_mask_abnormal = (true < self.ref_lower) | (true > self.ref_upper)
            mask = mask & ref_mask_abnormal

        true = true[mask]
        pred = pred[mask]

        try:
            normalized_mae, mae = self.normalized_mae(true, pred, min_v, max_v)
        except:
            for t, p in zip(true, pred):
                print(type(t), type(p), t, p)
            raise

        smape = self.smape(true, pred)
        return len(true), min_v, max_v, mae, normalized_mae, smape
    
    
    def calculate(self, path):
        pred_true_data = pickle.load(open(path, "rb"))
        
        total_df = []
        
        for item in pred_true_data:
            table = pred_true_data[item]
            if type(table) == list and type(table[0])==dict:
                table = pd.DataFrame(table)
            if 'prev' in table.columns:
                table['pred'] = table['prev']
                table = table.drop(columns=['prev'])
         
            true = list(table['true'].values)
            pred = list(table['pred'].values)

            count, min_v, max_v, mae, normalized_mae, smape = self.calculate_metrics(true, pred)
            
            total_df.append([item, count, min_v, max_v, mae, normalized_mae, smape])

        total_df = pd.DataFrame(total_df, columns=['item', 'count',  'min_v', 'max_v', 'mae', 'nmae', 'smape'])
        weighted_nmae = (total_df['count'] * total_df['nmae']).sum() / total_df['count'].sum()
        weighted_smape = (total_df['count'] * total_df['smape']).sum() / total_df['count'].sum()
        return weighted_nmae, weighted_smape
