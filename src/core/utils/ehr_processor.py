from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import torch
from datasets import Dataset
from transformers import AutoTokenizer
import json
import pickle
import re
import os
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import logging
from functools import partial
import multiprocessing
from collections import defaultdict
from omegaconf import DictConfig

from utils import get_tokenizer, time_to_minutes
from feature import MIMICIV, eICU, HIRID

logger = logging.getLogger(__name__)

class EHRBase(ABC):
    """Base class for EHR data sources."""
    @abstractmethod
    def __init__(self, cfg: DictConfig):
        pass

class EHRProcessorBase(ABC):
    """Base class for EHR processors."""
    def __init__(self, cfg: DictConfig, ehr_info: EHRBase):
        self.cfg = cfg
        self.ehr_info = ehr_info
        self.data_path = Path(cfg.data_path)
        self.dest_path = Path(cfg.dest_path) / f"{cfg.data_name}_{'_'.join(cfg.use_tables)}_{cfg.value_type}_{cfg.timeoffset}"
        self.dest_path.mkdir(parents=True, exist_ok=True)
        self.stayid_key = ehr_info.stayid_key
        self.patientid_key = ehr_info.patientid_key

    @abstractmethod
    def extract_data(self):
        """Extract raw EHR data and save to HDF5."""
        pass

    @abstractmethod
    def preprocess(self):
        """Preprocess EHR data for model training or inference."""
        pass

class EHRProcessor(EHRProcessorBase):
    """Processor for EHR datasets (MIMIC-IV, eICU, HIRID)."""
    def __init__(self, cfg: DictConfig, ehr_info: EHRBase):
        super().__init__(cfg, ehr_info)
        self.tokenizer = get_tokenizer(cfg)
        self.cohort = None
        self.num_bucket_dict = {} if cfg.value_type == 'num_bucket' else None
        self.day_tokens = torch.tensor([
            self.tokenizer.encode(f'[DAY_{i}]', add_special_tokens=False)[0]
            for i in range(1, cfg.data.max_day_len + 1)
        ])
        self.unique_tokens = torch.tensor([])

    def extract_data(self):
        """Extract and process raw EHR data, saving to HDF5."""
        self.cohort = self._set_cohort()
        total_table = self._process_tables()
        self._save_h5(total_table)
        self._make_cohort_split()

    def preprocess(self):
        """Preprocess EHR data for training or inference."""
        self._split_train_valid_test()
        for split in ['train', 'valid']:
            self._make_training_dataset(split)
        for split in ['valid', 'test']:
            self._make_inference_dataset(split)
        if self.num_bucket_dict:
            self._save_representative_values()

    def _set_cohort(self) -> pd.DataFrame:
        """Load and filter ICU stays based on minimum length of stay."""
        icu = pd.read_csv(self.data_path / self.ehr_info.icustay_fname)
        icu = self._make_compatible(icu)
        return icu[icu['los'] >= (self.cfg.min_los / 24)]

    def _make_compatible(self, icu: pd.DataFrame) -> pd.DataFrame:
        """Ensure ICU data compatibility across datasets."""
        return icu  # Override in subclasses for specific datasets

    def _process_tables(self) -> pd.DataFrame:
        """Process specified tables and concatenate results."""
        total_table = None
        for table_name in self.cfg.use_tables:
            logger.info(f"Processing table: {table_name}")
            table = self._process_table(table_name)
            table = self._process_columns(table, table_name)
            table_time_text = table[[self.stayid_key, 'time', 'text']]
            total_table = pd.concat([total_table, table_time_text], axis=0) if total_table is not None else table_time_text
        return total_table

    def _process_table(self, table_name: str) -> pd.DataFrame:
        """Process a single table."""
        table_info = self.ehr_info.table_candidates[table_name]
        table = pd.read_csv(self.data_path / table_info['fname'])
        table[table_info['timestamp']] = pd.to_datetime(table[table_info['timestamp']])
        table = self._table_specific_processing(table, table_name, table_info)
        table = self._process_time(table, table_info, table_info['timestamp'])
        table = self._code_to_description(table, table_info, table_info['item_col'])
        if self.cfg.select_items:
            table = self._select_freq_items(table, table_name)
        table.to_csv(self.dest_path / f"{table_name}_filtered.csv", index=False)
        return table

    def _table_specific_processing(self, table: pd.DataFrame, table_name: str, table_info: Dict) -> pd.DataFrame:
        """Apply dataset-specific processing to a table."""
        return table  # Override in subclasses

    def _process_time(self, table: pd.DataFrame, table_info: Dict, time_stamp_key: str) -> pd.DataFrame:
        """Process timestamps and compute time offsets."""
        if self.stayid_key in table.columns:
            table = table[[self.stayid_key, time_stamp_key] + table_info['use']].merge(
                self.cohort[[self.stayid_key, 'intime', 'outtime']], on=self.stayid_key, how='inner'
            )
        else:
            table = table[['hadm_id', time_stamp_key] + table_info['use']].merge(
                self.cohort[['hadm_id', self.stayid_key, 'intime', 'outtime']], on='hadm_id', how='inner'
            )
            table = table[table[self.stayid_key].isin(self.cohort[self.stayid_key])]
        table = table[(table[time_stamp_key] >= table['intime']) & (table[time_stamp_key] <= table['outtime'])]
        
        if self.cfg.timeoffset == 'abs':
            table['time'] = ((table[time_stamp_key] - table['intime']).dt.total_seconds() / 60).round(4)
        elif self.cfg.timeoffset == 'datetime':
            table['time'] = table[time_stamp_key]
            table['day_passed'] = (table[time_stamp_key].dt.date - table['intime'].dt.date).dt.days
        else:
            raise ValueError(f"Invalid timeoffset: {self.cfg.timeoffset}")
        
        return table.drop(columns=[time_stamp_key, 'intime', 'outtime'])

    def _code_to_description(self, table: pd.DataFrame, table_info: Dict, item_id: str) -> pd.DataFrame:
        """Convert codes to descriptive text."""
        if 'code' in table_info:
            table_desc = pd.read_csv(self.data_path / table_info['desc'])
            table = table.merge(table_desc[[table_info['code'], table_info['desc_key']]], on=table_info['code'], how='inner')
            table = table.rename(columns={table_info['desc_key']: 'item_&text'})
        else:
            table['item_&text'] = table[item_id]
        table = table.dropna(subset=['item_&text'])
        table['item_&text'] = table['item_&text'].str.lower()
        duplicate_id = item_id if self.cfg.drop_duplicates_by == 'itemid' else 'item_&text'
        table = table.drop_duplicates(subset=['time', self.stayid_key, duplicate_id], keep='first')
        return table.drop(columns=[item_id])

    def _select_freq_items(self, table: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Filter items by frequency."""
        if table_name == 'labevents' and self.cfg.use_10_labitems:
            return table
        item_freq = table['item_&text'].value_counts()
        total_items = item_freq.sum()
        threshold_freq = total_items * 0.9
        sorted_freq = item_freq.sort_values(ascending=False)
        cumsum_freq = sorted_freq.cumsum()
        min_freq = item_freq[cumsum_freq[cumsum_freq >= threshold_freq].index[0]]
        surviving_items = sorted_freq[sorted_freq >= min_freq].index.tolist()
        with open(self.dest_path / f"surviving_items_{table_name}.txt", 'w') as f:
            for item in surviving_items:
                f.write(f"{item} {item_freq[item]}\n")
        return table[table['item_&text'].isin(surviving_items)]

    def _save_h5(self, total_table: pd.DataFrame):
        """Save processed data to HDF5."""
        file_name = f"{self.cfg.data_name}.h5"
        with h5py.File(self.dest_path / file_name, 'w') as data_h5:
            ehr_g = data_h5.create_group('ehr')
            for stay_id, group in tqdm(total_table.groupby(self.stayid_key), desc="Saving HDF5"):
                group_lab = group[group['text'].str.contains(self.ehr_info.lab_table)]
                if len(group_lab) < 1:
                    continue
                group = group.sort_values(by='time')
                stay_g = ehr_g.create_group(str(stay_id))
                stay_g.create_dataset('time', data=group['time'].astype(str).values, compression="gzip")
                stay_g.create_dataset('text', data=group['text'].values, dtype=h5py.string_dtype(encoding='utf-8'), compression="gzip")
                stay_g.create_dataset('base_info', data=self._make_base_info(stay_id))

    def _make_base_info(self, stay_id: str) -> str:
        """Generate base information for a stay."""
        return ""  # Implement as needed

    def _make_cohort_split(self):
        """Split cohort into train, valid, test sets."""
        with h5py.File(self.dest_path / f"{self.cfg.data_name}.h5", 'r') as file:
            stay_ids = list(file['ehr'].keys())
        icu_patients = pd.read_csv(self.data_path / self.ehr_info.icustay_fname)
        cohort = icu_patients[icu_patients[self.stayid_key].astype(str).isin(stay_ids)]
        shuffled = cohort.groupby(self.patientid_key)[self.patientid_key].count().sample(frac=1, random_state=42)
        cum_len = shuffled.cumsum()
        total = sum(shuffled)
        cohort[f'split_42'] = 'train'
        cohort.loc[cohort[self.patientid_key].isin(shuffled[cum_len < int(total * 0.1)].index), 'split_42'] = 'test'
        cohort.loc[cohort[self.patientid_key].isin(shuffled[(cum_len >= int(total * 0.1)) & (cum_len < int(total * 0.2))].index), 'split_42'] = 'valid'
        cohort.to_csv(self.dest_path / f"{self.cfg.data_name}_cohort.csv", index=False)

    def _split_train_valid_test(self):
        """Load split indices from cohort."""
        cohort = pd.read_csv(self.dest_path / f"{self.cfg.data_name}_cohort.csv")
        self.test_idx = cohort[cohort['split_42'] == 'test'][self.stayid_key].astype(str).values
        self.valid_idx = cohort[cohort['split_42'] == 'valid'][self.stayid_key].astype(str).values
        self.train_idx = cohort[cohort['split_42'] == 'train'][self.stayid_key].astype(str).values

    def _make_training_dataset(self, split: str):
        """Create training dataset."""
        idx = {'train': self.train_idx, 'valid': self.valid_idx}[split]
        if self.cfg.mode.debugging_mode:
            idx = idx[:10]
        hf_dataset = defaultdict(list)
        with h5py.File(self.dest_path / f"{self.cfg.data_name}.h5", 'r') as h5_data:
            for icu_id in tqdm(idx, desc=f"Processing {split}"):
                ehr = h5_data['ehr'][icu_id]
                tokens, types = self._process_icu_events(ehr, split)
                if tokens is not None:
                    hf_dataset['icu_id'].append(icu_id)
                    hf_dataset['tokens'].append(tokens.to(torch.int32))
                    hf_dataset['types'].append(types.to(torch.int32))
                    self.unique_tokens = torch.cat((self.unique_tokens, tokens.to(torch.int32))).unique()
        dataset = Dataset.from_dict(hf_dataset)
        dataset.save_to_disk(self.dest_path / f"{split}_dataset_{self.cfg.max_seq_len}")

    def _make_inference_dataset(self, split: str):
        """Create inference dataset."""
        idx = {'test': self.test_idx, 'valid': self.valid_idx}[split]
        if self.cfg.mode.debugging_mode:
            idx = idx[:50]
        hf_dataset = defaultdict(list)
        with h5py.File(self.dest_path / f"{self.cfg.data_name}.h5", 'r') as h5_data:
            for icu_id in tqdm(idx, desc=f"Processing {split}"):
                ehr = h5_data['ehr'][icu_id]
                for event_data in self._process_icu_events_for_inference(ehr, split):
                    hf_dataset['icu_id'].append(icu_id)
                    hf_dataset.update(event_data)
                    if split == "test":
                        self.unique_tokens = torch.cat((self.unique_tokens, event_data['prompt_tokens'].to(torch.int32))).unique()
                        self.unique_tokens = torch.cat((self.unique_tokens, event_data['lab_tokens'].to(torch.int32))).unique()
        dataset = Dataset.from_dict(hf_dataset)
        dataset.save_to_disk(self.dest_path / f"{split}_prompt_dataset_{self.cfg.max_seq_len}")

    def _process_icu_events(self, ehr: h5py.Group, split: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Process ICU events for training."""
        base_info = ehr['base_info'][()].decode('utf-8') if self.cfg.data.base_info else ""
        base_tokens = torch.tensor(self.tokenizer.encode(base_info, add_special_tokens=False)) if base_info else torch.tensor([])
        base_types = torch.zeros_like(base_tokens)
        tokens, types = torch.tensor([]), torch.tensor([])
        prev_time = None
        for text in ehr['text']:
            event_text = eval(text.decode('utf-8'))
            if event_text[1] in ['labevents', 'lab', 'observation_tables'] and event_text[3] in ('_ _ _', '___'):
                continue
            event_toks, event_types = self._tokenize_event(event_text, split, prev_time)
            if not event_toks:
                continue
            total_len = len(base_tokens) + len(tokens) + len(event_toks)
            if total_len > self.cfg.max_seq_len:
                if len(tokens) > 0:
                    final_tokens = torch.cat((base_tokens, tokens)) if base_info else tokens
                    final_types = torch.cat((base_types, types)) if base_info else types
                    return final_tokens.to(torch.int32), final_types.to(torch.int32)
                tokens, types = torch.tensor([]), torch.tensor([])
            tokens = torch.cat((tokens, event_toks))
            types = torch.cat((types, event_types))
            prev_time = int(event_text[0]) if self.cfg.timeoffset == 'abs' else None
        if len(tokens) > 0:
            final_tokens = torch.cat((base_tokens, tokens)) if base_info else tokens
            final_types = torch.cat((base_types, types)) if base_info else types
            return final_tokens.to(torch.int32), final_types.to(torch.int32)
        return None, None

    def _process_icu_events_for_inference(self, ehr: h5py.Group, split: str) -> List[Dict]:
        """Process ICU events for inference."""
        base_info = ehr['base_info'][()].decode('utf-8') if self.cfg.data.base_info else ""
        base_tokens = torch.tensor(self.tokenizer.encode(base_info, add_special_tokens=False)) if base_info else torch.tensor([])
        base_types = torch.zeros_like(base_tokens)
        icu_tokens, icu_types = torch.tensor([]), torch.tensor([])
        item_recent_values = {}
        item_mean_prev_values = {}
        prev_time = None
        results = []
        for event_text in map(lambda x: eval(x.decode('utf-8')), ehr['text']):
            if event_text[1] in ['labevents', 'lab', 'observation_tables'] and event_text[3] in ('_ _ _', '___'):
                continue
            event_toks, event_types = self._tokenize_event(event_text, split, prev_time)
            if not event_toks:
                continue
            if event_text[1] in ['labevents', 'lab', 'observation_tables']:
                previous_value_tokens = item_recent_values.get(event_text[2], torch.tensor([]))
                prompt_tokens, prompt_types = self._remove_same_time_events(event_toks, icu_tokens, icu_types)
                length_fixed = len(base_tokens) + len(event_toks) if self.cfg.data.base_info else len(event_toks)
                if length_fixed + len(prompt_tokens) > self.cfg.max_seq_len:
                    prompt_tokens, prompt_types = self._cut_prompt(length_fixed, prompt_tokens, prompt_types)
                if len(prompt_tokens) > 0:
                    final_prompt = torch.cat((base_tokens, prompt_tokens)) if self.cfg.data.base_info else prompt_tokens
                    final_types = torch.cat((base_types, prompt_types)) if self.cfg.data.base_info else prompt_types
                    results.append({
                        'prompt_tokens': final_prompt.to(torch.int32),
                        'lab_tokens': event_toks.to(torch.int32),
                        'lab_types': event_types.to(torch.int32),
                        'lab_value': float(event_text[3].replace(' ', '')),
                        'previous_value_tokens': previous_value_tokens.to(torch.int32),
                        'mean_previous_values': item_mean_prev_values.get(event_text[2], [np.nan, np.nan])[1]
                    })
                try:
                    event_value = float(event_text[3].replace(' ', ''))
                    if event_text[2] not in item_mean_prev_values:
                        item_mean_prev_values[event_text[2]] = (1, event_value)
                    else:
                        len_, mean_ = item_mean_prev_values[event_text[2]]
                        new_mean = round((mean_ * len_ + event_value) / (len_ + 1), 2)
                        item_mean_prev_values[event_text[2]] = (len_ + 1, new_mean)
                    item_recent_values[event_text[2]] = torch.tensor(self.tokenizer.encode(' '.join(event_text[3:]), add_special_tokens=False))
                except ValueError:
                    logger.warning(f"Skipping non-numeric lab value for {event_text[2]}: {event_text[3]}")
            icu_tokens = torch.cat((icu_tokens, event_toks)).to(torch.int32)
            icu_types = torch.cat((icu_types, event_types)).to(torch.int32)
            prev_time = int(event_text[0]) if self.cfg.timeoffset == 'abs' else None
        return results

    def _tokenize_event(self, event_text: List, split: str, prev_time: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize an event."""
        event_toks, event_types = [], []
        table_text, item_text = event_text[1], event_text[2]
        for j, c in enumerate(event_text):
            if c not in ('', 'nan'):
                if self.cfg.timeoffset == 'abs' and j == 0:
                    time_gap = int(c) - prev_time if prev_time is not None else int(c)
                    c = f'[{time_gap}]'
                if self.cfg.data.num_bucket and self._is_convertible_to_float(c):
                    c = self._num_bucket(split, table_text, item_text, c)
                    if c is None:
                        return torch.tensor([]), torch.tensor([])
                toks = self.tokenizer.encode(c, add_special_tokens=False)
                event_toks.extend(toks)
                type_t = 1 if (table_text in ['labevents', 'lab', 'observation_tables'] and j > 2) else 0
                event_types.extend([type_t] * len(toks))
        if self.cfg.data.add_end_of_event:
            end_token = self.tokenizer.encode('|endofevent|', add_special_tokens=False)[0]
            event_toks.append(end_token)
            event_types.append(1 if table_text in ['labevents', 'lab', 'observation_tables'] else 0)
        return torch.tensor(event_toks), torch.tensor(event_types)

    def _is_convertible_to_float(self, s: str) -> bool:
        """Check if a string can be converted to float."""
        s = s.replace(' ', '')
        return bool(re.match(r'^[-+]?\d*\.?\d+(e[-+]?\d+)?$', s.strip()))

    def _num_bucket(self, split: str, table_text: str, item_text: str, value: str) -> Optional[str]:
        """Assign a value to a numerical bucket."""
        if not self.num_bucket_dict:
            return None
        key = (table_text, item_text)
        if key not in self.num_bucket_dict:
            logger.warning(f"No bucket for {key}")
            return None
        num_bucket_dict = self.num_bucket_dict[key]
        value = float(value.replace(' ', ''))
        for (min_val, max_val), bucket_id in num_bucket_dict.items():
            if min_val == max_val and min_val == value:
                return f'[NUM_{bucket_id}]'
            if min_val <= value < max_val or (min_val < value <= max_val and max_val == list(num_bucket_dict.keys())[-1][1]):
                return f'[NUM_{bucket_id}]'
        logger.warning(f"No bucket found for {key}, value: {value}")
        return None

    def _save_representative_values(self):
        """Save representative values for numerical buckets."""
        # Implement as needed
        pass

    def _remove_same_time_events(self, event_toks: torch.Tensor, tokens: torch.Tensor, types: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove events with the same timestamp."""
        num_time_tokens = 4 if self.cfg.data.add_weekday else 3
        if len(tokens) < num_time_tokens:
            return tokens, types
        time_tokens = torch.tensor(event_toks[:num_time_tokens])
        match_indices = (tokens.unfold(0, num_time_tokens, 1) == time_tokens).all(dim=1).nonzero(as_tuple=True)[0]
        if len(match_indices) > 0:
            first_match_idx = match_indices[0].item()
            return tokens[:first_match_idx], types[:first_match_idx]
        return tokens, types

    def _cut_prompt(self, length_fixed: int, tokens: torch.Tensor, types: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cut prompt to fit within max sequence length."""
        seq_len = self.cfg.max_seq_len - length_fixed
        tokens = tokens[-seq_len:]
        types = types[-seq_len:]
        first_day_idx = torch.where(torch.isin(tokens, self.day_tokens))[0][0]
        return tokens[first_day_idx:], types[first_day_idx:]

class MIMICProcessor(EHRProcessor):
    """Processor for MIMIC-IV dataset."""
    def _table_specific_processing(self, table: pd.DataFrame, table_name: str, table_info: Dict) -> pd.DataFrame:
        if table_name == 'labevents' and self.cfg.use_10_labitems:
            table = table[table['itemid'].isin(table_info['itemid'])]
            table = table[table['valuenum'].notnull()]
            table['value'] = table['valuenum']
            table = table.drop(columns=['valuenum'])
        elif table_name == 'emar':
            table_detail = pd.read_csv(self.data_path / table_info['detail'])
            table_detail = table_detail[table_detail['dose_given'].notnull() & (table_detail['dose_given'] != '___')]
            table = table.merge(table_detail, on='emar_id', how='left')
            table = table.rename(columns={table_info['item_col']: 'medication'})
        return table

class EICUProcessor(EHRProcessor):
    """Processor for eICU dataset."""
    def _make_compatible(self, icu: pd.DataFrame) -> pd.DataFrame:
        icu['los'] = icu['unitdischargeoffset'] / 60 / 24
        icu = icu.dropna(subset=['age'])
        icu = icu.rename(columns={'unitadmittime24': 'intime', 'unitdischargeoffset': 'outtime'})
        return icu

    def _table_specific_processing(self, table: pd.DataFrame, table_name: str, table_info: Dict) -> pd.DataFrame:
        if table_name == 'lab':
            table['labmeasurenamesystem'] = table['labmeasurenamesystem'].fillna(table['labmeasurenameinterface'])
            table = table.drop(columns=['labmeasurenameinterface'])
        elif table_name == 'medication':
            table['dosage'] = table['dosage'].str.replace(' ', ', ', regex=False)
        return table

class HIRIDProcessor(EHRProcessor):
    """Processor for HIRID dataset."""
    def _set_cohort(self) -> pd.DataFrame:
        self._extract_icu()
        icu = pd.read_csv(self.data_path / self.ehr_info.icustay_fname)
        icu = self._make_compatible(icu)
        return icu[icu['los'] >= (self.cfg.min_los / 24)]

    def _extract_icu(self):
        """Extract HIRID data from tar files."""
        # Implement tar extraction as needed
        pass

    def _make_compatible(self, icu: pd.DataFrame) -> pd.DataFrame:
        icu = icu.rename(columns={'admissiontime': 'intime'})
        icu['intime'] = pd.to_datetime(icu['intime'])
        dfs = []
        for table_name, table_info in self.ehr_info.table_candidates.items():
            df = pd.read_parquet(self.data_path / table_info['fname'])
            df = df[[table_info['timestamp'], self.stayid_key]]
            df[table_info['timestamp']] = pd.to_datetime(df[table_info['timestamp']])
            dfs.append(df.rename(columns={table_info['timestamp']: 'timestamp'}))
        merged_df = pd.concat(dfs, ignore_index=True).groupby(self.stayid_key).agg(max_time=('timestamp', 'max')).reset_index()
        icu = icu.merge(merged_df, on=self.stayid_key, how='inner')
        icu['los'] = (icu['max_time'] - icu['intime']).dt.total_seconds() / 60 / 60 / 24
        icu.to_csv(self.data_path / 'icustays.csv', index=False)
        return icu

    def _table_specific_processing(self, table: pd.DataFrame, table_name: str, table_info: Dict) -> pd.DataFrame:
        if table_name == 'observation_tables':
            # Handle specific processing
            pass
        return table

class EHRProcessorFactory:
    """Factory for creating EHR processors."""
    _processor_types = {
        'mimiciv': MIMICProcessor,
        'eicu': EICUProcessor,
        'hirid': HIRIDProcessor
    }

    @classmethod
    def create(cls, data_name: str, cfg: DictConfig, ehr_info: EHRBase) -> EHRProcessorBase:
        processor_class = cls._processor_types.get(data_name.lower())
        if not processor_class:
            raise ValueError(f"Unsupported EHR data source: {data_name}")
        return processor_class(cfg, ehr_info)