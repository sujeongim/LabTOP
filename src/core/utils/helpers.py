from transformers import AutoTokenizer
from omegaconf import DictConfig
import re
import os
import pickle
from data.dataset import EHRGPTDataset, PromptTestDataset

def ensure_dir(path):
    """
    Ensure that a directory exists.
    If it doesn't exist, create it.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    special_tokens = build_special_tokens(cfg)
    
    if special_tokens:
        num_added = tokenizer.add_special_tokens({
            "additional_special_tokens": special_tokens
        })
        print("We have added", num_added, "tokens\n")
    
    return tokenizer

def build_special_tokens(cfg):
    tokens = ["[unused1]"]
    
    if cfg.data.time_gap == "token":
        tokens += [f"[unused{i}]" for i in range(2, 12)]
    
    tokens += [f'[NUM_{i}]' for i in range(1, 6)]

    if cfg.data.timeoffset == 'datetime':
        tokens += build_datetime_tokens(cfg)

    if cfg.data.num_bucket:
        if cfg.data.num_bucket_per_item:
            tokens += build_bucket_tokens_per_item(cfg)
        elif cfg.data.num_bucket_num > 5:
            tokens += [f'[NUM_{i}]' for i in range(6, cfg.data.num_bucket_num + 1)]

    tokens.append('|endofevent|')
    return tokens

def build_datetime_tokens(cfg):
    date_tokens = [f'[DAY_{d}]' for d in range(1, cfg.data.max_day_len + 1)]

    if cfg.data.add_weekday:
        date_tokens += ['[MON]', '[TUE]', '[WED]', '[THU]', '[FRI]', '[SAT]', '[SUN]']

    date_tokens += [f'[{h:02}h]' for h in range(24)]
    date_tokens += [f'[{m:02}m]' for m in range(0, 60, 10)]
    
    return date_tokens

def build_bucket_tokens_per_item(cfg):
    path = os.path.join(cfg.data_path, f"mimiciv_{cfg.data.num_bucket_num}_percentile_buckets.pkl")
    with open(path, "rb") as f:
        num_bucket_dict = pickle.load(f)

    bucket_tokens = []
    for (k1, k2), v in num_bucket_dict.items():
        prefix = k1.split("events")[0] + '_' + k2
        for i in range(len(set(v)) - 1):
            bucket_tokens.append(f'[{prefix}_{i + 1}]')
    return bucket_tokens

def time_to_minutes(time_str: str) -> int:
    """Convert a time string (e.g., '01:02:03') into total minutes."""
    h, m, s = map(int, re.findall(r'\d+', time_str))
    return h * 60 * 24 + m * 60 + s

def make_dataset(cfg, tokenizer, prompt_test=False):
    data_path = cfg.data_path

    if prompt_test:
        train_dataset = valid_dataset = None
        test_dataset = load_test_dataset(cfg, tokenizer, data_path, prompt_test)
    else:
        print('Get data file')
        train_dataset = EHRGPTDataset(cfg, tokenizer, os.path.join(data_path, f'train_dataset_{cfg.max_seq_len}'), train=True)
        valid_dataset = EHRGPTDataset(cfg, tokenizer, os.path.join(data_path, f'valid_dataset_{cfg.max_seq_len}'), train=False)
        test_dataset = load_validation_or_none(cfg, tokenizer, data_path)

    return train_dataset, valid_dataset, test_dataset, None

def load_test_dataset(cfg, tokenizer, data_path, prompt_test):
    filename = f'test_prompt_dataset_{cfg.max_seq_len}'
    if cfg.test.sampled:
        filename += '_sampled_llm'
        print('Get sampled test file')
    return PromptTestDataset(cfg, tokenizer, os.path.join(data_path, filename), prompt_test)

def load_validation_or_none(cfg, tokenizer, data_path):
    if cfg.train.best_model_criteria == "value_acc":
        return PromptTestDataset(cfg, tokenizer, os.path.join(data_path, f'valid_prompt_dataset_{cfg.max_seq_len}'), prompt_test=False)
    return None




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
def set_seed(seed=42):
    """seed setting 함수"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    
def calculate_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_optimal_num_workers(batch_size, max_workers=None, multiplier=1.0, min_workers=1):
    """
    Determine the optimal number of workers for parallel tasks based on batch size.

    Parameters:
        batch_size (int): The batch size for the workload.
        max_workers (int): Maximum number of workers to use. If None, no upper limit is set.
        multiplier (float): Factor to adjust the number of workers relative to cores. Default is 1.0.
        min_workers (int): Minimum number of workers to return. Default is 1.

    Returns:
        int: The optimal number of workers.
    """
    try:
        # Number of CPU cores
        num_cores = multiprocessing.cpu_count()
        
        # Initial workers based on cores and multiplier
        optimal_workers = int(num_cores * multiplier)
        
        # Adjust workers based on batch size
        # Assume a rough heuristic: workers ~ sqrt(batch_size)
        batch_based_workers = max(int(batch_size ** 0.5), 1)
        
        # Final workers: the minimum of system resources and batch requirements
        optimal_workers = min(optimal_workers, batch_based_workers)
        
        # Cap workers by max_workers if provided
        if max_workers is not None:
            optimal_workers = min(optimal_workers, max_workers)
        
        # Ensure minimum number of workers
        return max(optimal_workers, min_workers)
    except Exception as e:
        # Fallback in case of an error
        print(f"Error determining the number of workers: {e}")
        return min_workers
    
def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)

    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)

    else:
        dirs = glob.glob(f"{path}*")
        matches = [int(d.split('_')[-1]) for d in dirs if d.split('_')[-1].isdecimal()]
        #print(i)
        n = max(matches) + 1 if matches else 1
        return f"{path}_{n}"


class EarlyStopping(object):
    """
    https://wandb.ai/ayush-thakur/huggingface/reports/Early-Stopping-in-HuggingFace-Examples--Vmlldzo0MzE2MTM
    """

    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = float('inf')
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if math.isinf(self.best):
            self.best = metrics
            return False

        if math.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print('terminating because of early stopping!')
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                    best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                    best * min_delta / 100)
