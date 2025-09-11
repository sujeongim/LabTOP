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
import time
import threading
import psutil
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import logging
from functools import partial
import multiprocessing
from collections import defaultdict
from omegaconf import DictConfig
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import time
import psutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.utils.helpers import get_tokenizer, time_to_minutes
from core.utils.feature import MIMICIV, eICU, HIRID

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
        self.dest_path = Path(cfg.dest_path) / f"{cfg.data_name}_{'_'.join(cfg.data.use_tables)}_{cfg.data.value_type}_{cfg.data.timeoffset}"
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
        self.num_bucket_dict = {} if cfg.data.value_type == 'num_bucket' else None
        self.day_tokens = torch.tensor([
            self.tokenizer.encode(f'[DAY_{i}]', add_special_tokens=False)[0]
            for i in range(1, cfg.data.max_day_len + 1)
        ])
        
        # 요일 토큰 생성
        self.weekday_tokens = torch.tensor([
            self.tokenizer.encode(f'[{day}]', add_special_tokens=False)[0]
            for day in ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
        ])
        
        # 시간 토큰 생성 (00h~23h)
        self.hour_tokens = torch.tensor([
            self.tokenizer.encode(f'[{h:02d}h]', add_special_tokens=False)[0]
            for h in range(24)
        ])
        
        # 분 토큰 생성 (10분 단위: 00m, 10m, 20m, 30m, 40m, 50m)
        self.minute_tokens = torch.tensor([
            self.tokenizer.encode(f'[{m:02d}m]', add_special_tokens=False)[0]
            for m in range(0, 60, 10)
        ])
        
        # 시간 관련 토큰 통합
        time_tokens = torch.cat([self.day_tokens, self.weekday_tokens, self.hour_tokens, self.minute_tokens])
        self.unique_tokens = time_tokens
        
        # 병렬화 설정
        self.n_workers = getattr(cfg, 'n_workers', min(multiprocessing.cpu_count(), 8))
        self.chunk_size = getattr(cfg, 'chunk_size', 1000)
        self._lock = threading.Lock()
        
        # 성능 모니터링
        self.start_time = None
        self.memory_usage = []
        
        # 샘플 값 로깅 추적 (각 컬럼당 한 번만 로깅)
        self._logged_columns = set()
        
        # 디버깅 모드일 때 DEBUG 레벨로 설정
        if getattr(cfg, 'debugging_mode', False):
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
            
        self.use_tables = sorted(cfg.data.use_tables)

    def extract_data(self):
        """Extract and process raw EHR data, saving to HDF5."""
        self.start_time = time.time()
        self._log_memory_usage("Starting extract_data")
        
        self.cohort = self._set_cohort()
        self._log_memory_usage("After setting cohort")
        
        total_table = self._process_tables()
        self._log_memory_usage("After processing tables")
        
        self._save_h5(total_table)
        self._log_memory_usage("After saving HDF5")
        
        self._make_cohort_split()
        self._log_memory_usage("After making cohort split")
        
        elapsed_time = time.time() - self.start_time
        logger.info(f"=== extract_data completed in {elapsed_time:.2f} seconds ===")

    def preprocess(self):
        """Preprocess EHR data for training or inference."""
        self.start_time = time.time()
        self._log_memory_usage("Starting preprocess")
        
        self._split_train_valid_test()
        self._log_memory_usage("After splitting train/valid/test")
        
        # Training datasets
        for split in ['train', 'valid']:
            # if there is no training data
            target_dir = self.dest_path / f"{split}_dataset_{self.cfg.max_seq_len}"
            if target_dir.exists():
                logger.info(f"Skipping {split} dataset creation, already exists: {target_dir}")
                continue
            self._make_training_dataset(split)
            self._log_memory_usage(f"After making {split} training dataset")
        
        # Inference datasets
        for split in ['valid', 'test']:
            self._make_inference_dataset(split)
            self._log_memory_usage(f"After making {split} inference dataset")
        
        if self.num_bucket_dict:
            self._save_representative_values()
        
        elapsed_time = time.time() - self.start_time
        logger.info(f"=== preprocess completed in {elapsed_time:.2f} seconds ===")
        
        # 전체 성능 요약 출력
        self._print_performance_summary()

    def _set_cohort(self) -> pd.DataFrame:
        """Load and filter ICU stays based on minimum length of stay."""
        icu = pd.read_csv(self.data_path / self.ehr_info.icustay_fname)
        icu = self._make_compatible(icu)
        return icu[icu['los'] >= (self.cfg.data.min_los / 24)]

    def _make_compatible(self, icu: pd.DataFrame) -> pd.DataFrame:
        """Ensure ICU data compatibility across datasets."""
        return icu  # Override in subclasses for specific datasets

    def _process_tables(self) -> pd.DataFrame:
        """Process specified tables and concatenate results with parallel processing."""
        def process_single_table(table_name):
            """단일 테이블 처리"""
            logger.info(f"Processing table: {table_name}")
            table = self._process_table(table_name)
            
            # 테이블이 비어있으면 None 반환
            if table.empty:
                logger.warning(f"Table '{table_name}' is empty, skipping...")
                return None
            
            # value와 valueuom 컬럼 포함하여 반환
            required_cols = [self.stayid_key, 'time', 'tablename_text', 'itemname_text']
            optional_cols = ['value', 'valueuom']
            
            # 존재하는 컬럼만 포함
            available_cols = [col for col in required_cols + optional_cols if col in table.columns]
            return table[available_cols]
        
        # 병렬로 테이블 처리
        logger.info(f"Processing {len(self.use_tables)} tables with {self.n_workers} workers")
        table_results = []
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(process_single_table, table_name) for table_name in self.use_tables]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tables"):
                table_results.append(future.result())
        
        # 결과 병합
        total_table = None
        for table_time_text in table_results:
            if table_time_text is not None and len(table_time_text) > 0:
                total_table = pd.concat([total_table, table_time_text], axis=0) if total_table is not None else table_time_text
        
        # 데이터가 비어있는지 확인
        if total_table is None or len(total_table) == 0:
            logger.error("No data found after processing tables. Please check your data and configuration.")
            raise ValueError("No data found after processing tables")
        
        # ICU intime 정보와 merge
        logger.info("Merging with ICU intime information...")
        cohort_intime = self.cohort[[self.stayid_key, 'intime']].copy()
        total_table = total_table.merge(cohort_intime, on=self.stayid_key, how='inner')
        
        # intime이 없는 데이터 필터링
        total_table = total_table.dropna(subset=['intime'])
        
        # datetime 타입으로 변환
        total_table = self._ensure_datetime(total_table, 'intime')
        
        logger.info(f"Total processed data shape after ICU intime merge: {total_table.shape}")
        return total_table

    def _process_table(self, table_name: str) -> pd.DataFrame:
        """Process a single table with memory optimization."""
        table_info = self.ehr_info.table_candidates[table_name]
        
        # 청크 단위로 CSV 읽기 (대용량 파일 처리 최적화)
        chunk_size = 100000  # 메모리 사용량 조절
        table_chunks = []
        
        # debugging 모드일 때 샘플링 비율 설정
        sample_ratio = getattr(self.cfg.data, 'debug_table_sample_ratio', 0.1) if getattr(self.cfg, 'debugging_mode', False) else 1.0
        
        logger.info(f"Processing table '{table_name}' with sample ratio: {sample_ratio}")
        
        for chunk in pd.read_csv(self.data_path / table_info['fname'], chunksize=chunk_size, low_memory=False):
            # debugging 모드일 때 청크의 일부만 샘플링
            if sample_ratio < 1.0:
                original_size = len(chunk)
                chunk = chunk.sample(frac=sample_ratio, random_state=42)
                logger.info(f"Sampled {len(chunk)}/{original_size} rows from chunk in table '{table_name}'")
            
            # 안전한 datetime 변환
            chunk = self._ensure_datetime(chunk, table_info['timestamp'])
            
            # datetime 변환 실패한 행 처리
            failed_conversion = chunk[table_info['timestamp']].isna()
            if failed_conversion.any():
                failed_count = failed_conversion.sum()
                total_count = len(chunk)
                logger.warning(f"Chunk: {failed_count}/{total_count} rows have invalid datetime in '{table_info['timestamp']}'")
                
                # 설정에 따라 실패한 행 제거 또는 경고만 출력
                if hasattr(self.cfg, 'skip_invalid_datetime') and self.cfg.skip_invalid_datetime:
                    chunk = chunk.dropna(subset=[table_info['timestamp']])
                    logger.info(f"Removed {failed_count} rows with invalid datetime")
                else:
                    # NaN 값이 있는 행 제거 (기본 동작)
                    chunk = chunk.dropna(subset=[table_info['timestamp']])
            
            chunk = self._table_specific_processing(chunk, table_name, table_info)
            
            # 단계별 데이터 추적을 위한 로깅
            chunk_after_specific = len(chunk)
            logger.debug(f"After table-specific processing: {chunk_after_specific} rows")
            
            chunk = self._process_time(chunk, table_info, table_info['timestamp'])
            chunk_after_time = len(chunk)
            logger.debug(f"After time processing: {chunk_after_time} rows")
            
            chunk = self._code_to_description(chunk, table_info, table_info['item_col'], table_name)
            chunk_after_code = len(chunk)
            logger.debug(f"After code to description: {chunk_after_code} rows")
            
            if self.cfg.data.select_items:
                chunk = self._select_freq_items(chunk, table_name)
                chunk_after_freq = len(chunk)
                logger.debug(f"After frequency filtering: {chunk_after_freq} rows")
            
            table_chunks.append(chunk)
        
        # 청크들을 병합
        if not table_chunks:
            logger.warning(f"No data found for table '{table_name}'")
            return pd.DataFrame()
        
        table = pd.concat(table_chunks, ignore_index=True)
        
        # 데이터가 비어있는지 확인
        if len(table) == 0:
            logger.warning(f"Table '{table_name}' is empty after processing")
            return pd.DataFrame()
        
        logger.info(f"Table '{table_name}' processed: {len(table)} rows")
        table.to_csv(self.dest_path / f"{table_name}_filtered.csv", index=False)
        return table

    def _table_specific_processing(self, table: pd.DataFrame, table_name: str, table_info: Dict) -> pd.DataFrame:
        """Apply dataset-specific processing to a table."""
        return table  # Override in subclasses

    def _process_time(self, table: pd.DataFrame, table_info: Dict, time_stamp_key: str) -> pd.DataFrame:
        """Process timestamps and compute time information including datetime components."""
        if self.stayid_key in table.columns:
            table = table[[self.stayid_key, time_stamp_key] + table_info['use']].merge(
                self.cohort[[self.stayid_key, 'intime', 'outtime']], on=self.stayid_key, how='inner'
            )
        else:
            table = table[['hadm_id', time_stamp_key] + table_info['use']].merge(
                self.cohort[['hadm_id', self.stayid_key, 'intime', 'outtime']], on='hadm_id', how='inner'
            )
            table = table[table[self.stayid_key].isin(self.cohort[self.stayid_key])]
        
        # datetime 타입 확인 및 변환
        table = self._ensure_datetime(table, time_stamp_key)
        table = self._ensure_datetime(table, 'intime')
        table = self._ensure_datetime(table, 'outtime')
        
        # NaN 값 제거
        table = table.dropna(subset=[time_stamp_key, 'intime', 'outtime'])
        
        # 필터링 전 데이터 상태 로깅
        before_count = len(table)
        if before_count > 0:
            logger.debug(f"Before time filtering: {before_count} rows")
            # 샘플 데이터 로깅
            sample_data = table.head(5)
            for idx, row in sample_data.iterrows():
                logger.debug(f"  Sample row {idx}: {row[time_stamp_key]} (intime: {row['intime']}, outtime: {row['outtime']})")
        
        # 시간 조건 확인
        time_condition = (table[time_stamp_key] >= table['intime']) & (table[time_stamp_key] <= table['outtime'])
        valid_time_mask = time_condition
        invalid_time_count = (~valid_time_mask).sum()
        
        if invalid_time_count > 0:
            logger.warning(f"Found {invalid_time_count}/{before_count} rows with timestamps outside ICU stay period")
            # 잘못된 시간 데이터의 샘플 출력
            invalid_sample = table[~valid_time_mask].head(3)
            for idx, row in invalid_sample.iterrows():
                logger.warning(f"  Invalid time row {idx}: {row[time_stamp_key]} not in [{row['intime']}, {row['outtime']}]")
        
        table = table[valid_time_mask]
        
        after_count = len(table)
        logger.info(f"Time filtering: {before_count} -> {after_count} rows (removed {before_count - after_count})")
        if before_count > 0:
            retention_rate = (after_count / before_count) * 100
            logger.info(f"Time filtering retention rate: {retention_rate:.2f}%")
        
        # 시간 정보를 별도 컬럼으로 저장 (토큰화에 사용)
        table['event_datetime'] = table[time_stamp_key]
        table['icu_intime'] = table['intime']
        
        if self.cfg.data.timeoffset == 'abs':
            try:
                table['time'] = ((table[time_stamp_key] - table['intime']).dt.total_seconds() / 60).round(4)
            except AttributeError as e:
                logger.warning(f"Error calculating absolute time offset: {e}")
                # fallback: 직접 계산
                time_diff = table[time_stamp_key] - table['intime']
                table['time'] = (time_diff.dt.total_seconds() / 60).round(4)
        elif self.cfg.data.timeoffset == 'datetime':
            table['time'] = table[time_stamp_key]
            # 안전한 날짜 계산
            try:
                table['day_passed'] = (table[time_stamp_key].dt.date - table['intime'].dt.date).dt.days
            except AttributeError:
                # fallback: 직접 계산
                table['day_passed'] = (table[time_stamp_key] - table['intime']).dt.days
        else:
            raise ValueError(f"Invalid data.timeoffset: {self.cfg.data.timeoffset}")
        
        return table.drop(columns=[time_stamp_key, 'intime', 'outtime'])

    def _code_to_description(self, table: pd.DataFrame, table_info: Dict, item_id: str, table_name: str) -> pd.DataFrame:
        """Convert codes to descriptive text and preserve value information."""
        if 'code' in table_info:
            table_desc = pd.read_csv(self.data_path / table_info['desc'])
            table = table.merge(table_desc[[table_info['code'], table_info['desc_key']]], on=table_info['code'], how='inner')
            table = table.rename(columns={table_info['desc_key']: 'itemname_text'})
        else:
            table['itemname_text'] = table[item_id]
        table = table.dropna(subset=['itemname_text'])
        table['itemname_text'] = table['itemname_text'].str.lower()
        
        # 테이블 이름을 별도 컬럼으로 추가
        table['tablename_text'] = table_name
        
        # value와 valueuom 컬럼 처리 (있는 경우)
        value_cols = []
        if 'value' in table_info.get('use', []):
            # value 컬럼이 table_info의 use에 정의되어 있는 경우
            value_col = table_info['use'][table_info['use'].index('value')]
            if value_col in table.columns:
                table['value'] = table[value_col].astype(str)
                value_cols.append('value')
        elif 'valuenum' in table.columns:
            # MIMIC-IV의 경우 valuenum 사용
            table['value'] = table['valuenum'].astype(str)
            value_cols.append('value')
        elif 'labresult' in table.columns:
            # eICU의 경우 labresult 사용
            table['value'] = table['labresult'].astype(str)
            value_cols.append('value')
        
        if 'valueuom' in table.columns:
            table['valueuom'] = table['valueuom'].astype(str)
            value_cols.append('valueuom')
        elif 'labmeasurenamesystem' in table.columns:
            # eICU의 경우 unit 정보
            table['valueuom'] = table['labmeasurenamesystem'].astype(str)
            value_cols.append('valueuom')
        
        duplicate_id = item_id if self.cfg.drop_duplicates_by == 'itemid' else 'itemname_text'
        
        # 중복 제거 전 데이터 상태 로깅
        before_dedup_count = len(table)
        logger.debug(f"Before duplicate removal: {before_dedup_count} rows")
        
        table = table.drop_duplicates(subset=['time', self.stayid_key, duplicate_id], keep='first')
        
        after_dedup_count = len(table)
        removed_duplicates = before_dedup_count - after_dedup_count
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows (kept {after_dedup_count}/{before_dedup_count})")
            retention_rate = (after_dedup_count / before_dedup_count) * 100
            logger.info(f"Duplicate removal retention rate: {retention_rate:.2f}%")
        
        # item_id 컬럼 제거 (불필요한 컬럼들도 함께)
        cols_to_drop = [item_id]
        # 원본 value 관련 컬럼들 제거 (새로 생성한 value, valueuom은 유지)
        if 'valuenum' in table.columns and 'value' in value_cols:
            cols_to_drop.append('valuenum')
        if 'labresult' in table.columns and 'value' in value_cols:
            cols_to_drop.append('labresult')
        if 'labmeasurenamesystem' in table.columns and 'valueuom' in value_cols:
            cols_to_drop.append('labmeasurenamesystem')
        
        # 존재하는 컬럼만 제거
        cols_to_drop = [col for col in cols_to_drop if col in table.columns]
        if cols_to_drop:
            table = table.drop(columns=cols_to_drop)
        
        return table

    def _select_freq_items(self, table: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Filter items by frequency."""
        if table_name == 'labevents' and getattr(self.cfg, 'use_10_labitems', False):
            return table
        
        before_freq_filter_count = len(table)
        logger.debug(f"Before frequency filtering: {before_freq_filter_count} rows")
        
        item_freq = table['itemname_text'].value_counts()
        total_items = item_freq.sum()
        threshold_freq = total_items * 0.9
        sorted_freq = item_freq.sort_values(ascending=False)
        cumsum_freq = sorted_freq.cumsum()
        min_freq = item_freq[cumsum_freq[cumsum_freq >= threshold_freq].index[0]]
        surviving_items = sorted_freq[sorted_freq >= min_freq].index.tolist()
        
        logger.info(f"Frequency filtering for {table_name}: keeping {len(surviving_items)} items out of {len(item_freq)} unique items")
        logger.info(f"Minimum frequency threshold: {min_freq} (90% cumulative frequency)")
        
        with open(self.dest_path / f"surviving_items_{table_name}.txt", 'w') as f:
            for item in surviving_items:
                f.write(f"{item} {item_freq[item]}\n")
        
        filtered_table = table[table['itemname_text'].isin(surviving_items)]
        after_freq_filter_count = len(filtered_table)
        removed_by_freq = before_freq_filter_count - after_freq_filter_count
        
        if removed_by_freq > 0:
            logger.info(f"Frequency filtering removed {removed_by_freq} rows (kept {after_freq_filter_count}/{before_freq_filter_count})")
            retention_rate = (after_freq_filter_count / before_freq_filter_count) * 100
            logger.info(f"Frequency filtering retention rate: {retention_rate:.2f}%")
        
        return filtered_table

    def _generate_time_tokens(self, event_datetime: pd.Timestamp, icu_intime: pd.Timestamp) -> List[str]:
        """Generate time tokens: [DAY_d][WEEKDAY][HHh][MMm]"""
        # ICU 입원 시점부터 지난 일수 계산
        days_passed = (event_datetime.date() - icu_intime.date()).days + 1  # 1부터 시작
        days_passed = max(1, min(days_passed, self.cfg.data.max_day_len))  # 범위 제한
        
        # 요일 (0: 월요일, 6: 일요일)
        weekday_names = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
        weekday = weekday_names[event_datetime.weekday()]
        
        # 시간 (0-23)
        hour = event_datetime.hour
        
        # 분 (10분 단위로 반올림: 0, 10, 20, 30, 40, 50)
        minute = (event_datetime.minute // 10) * 10
        
        return [
            f'[DAY_{days_passed}]',
            f'[{weekday}]', 
            f'[{hour:02d}h]',
            f'[{minute:02d}m]'
        ]

    def _save_h5(self, total_table: pd.DataFrame):
        """Save processed data to HDF5 with parallel processing."""
        file_name = f"{self.cfg.data_name}.h5"
        
        # 데이터를 청크로 나누기
        stay_groups = list(total_table.groupby(self.stayid_key))
        chunks = [stay_groups[i:i + self.chunk_size] for i in range(0, len(stay_groups), self.chunk_size)]
        
        # 임시 파일들을 저장할 디렉토리
        temp_dir = self.dest_path / "temp_h5"
        temp_dir.mkdir(exist_ok=True)
        
        def process_chunk(chunk_data, chunk_idx):
            """청크 단위로 HDF5 데이터 처리"""
            temp_file = temp_dir / f"chunk_{chunk_idx}.h5"
            processed_stays = []
            
            with h5py.File(temp_file, 'w') as temp_h5:
                ehr_g = temp_h5.create_group('ehr')
                for stay_id, group in chunk_data:
                    # 데이터가 비어있지 않은지 확인
                    if len(group) == 0:
                        continue
                    
                    group = group.sort_values(by='time')
                    stay_g = ehr_g.create_group(str(stay_id))
                    
                    # 기본 데이터 저장
                    stay_g.create_dataset('time', data=group['time'].astype(str).values, compression="gzip")
                    stay_g.create_dataset('tablename', data=group['tablename_text'].values, dtype=h5py.string_dtype(encoding='utf-8'), compression="gzip")
                    stay_g.create_dataset('itemname', data=group['itemname_text'].values, dtype=h5py.string_dtype(encoding='utf-8'), compression="gzip")
                    
                    # datetime 정보 저장 (시간 토큰 생성용)
                    #breakpoint()
                    if 'time' in group.columns and 'intime' in group.columns:
                        event_datetimes = []
                        icu_intimes = []
                        for _, row in group.iterrows():
                            event_datetimes.append(str(row['time']))
                            icu_intimes.append(str(row['intime']))
                        
                        stay_g.create_dataset('event_datetime', data=event_datetimes, dtype=h5py.string_dtype(encoding='utf-8'), compression="gzip")
                        stay_g.create_dataset('icu_intime', data=icu_intimes, dtype=h5py.string_dtype(encoding='utf-8'), compression="gzip")
                    
                    # value 정보 저장 (있는 경우)
                    if 'value' in group.columns:
                        values = group['value'].fillna('').astype(str).values
                        stay_g.create_dataset('value', data=values, dtype=h5py.string_dtype(encoding='utf-8'), compression="gzip")
                    
                    # valueuom 정보 저장 (있는 경우)
                    if 'valueuom' in group.columns:
                        valueuoms = group['valueuom'].fillna('').astype(str).values
                        stay_g.create_dataset('valueuom', data=valueuoms, dtype=h5py.string_dtype(encoding='utf-8'), compression="gzip")
                    
                    stay_g.create_dataset('base_info', data=self._make_base_info(stay_id))
                    processed_stays.append(stay_id)
            
            if not processed_stays:
                logger.warning(f"No valid stays found in chunk {chunk_idx}")
                return temp_file, []
            
            return temp_file, processed_stays
        
        # 병렬로 청크 처리
        logger.info(f"Processing {len(chunks)} chunks with {self.n_workers} workers")
        chunk_results = []
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(process_chunk, chunk, i) for i, chunk in enumerate(chunks)]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                chunk_results.append(future.result())
        
        # 결과를 메인 파일로 병합
        logger.info("Merging chunk files into main HDF5 file")
        
        # 처리된 데이터가 있는지 확인
        total_processed_stays = sum(len(processed_stays) for _, processed_stays in chunk_results)
        if total_processed_stays == 0:
            logger.error("No valid data found after processing. Please check your data and configuration.")
            raise ValueError("No valid data found after processing")
        
        logger.info(f"Total processed stays: {total_processed_stays}")
        
        with h5py.File(self.dest_path / file_name, 'w') as main_h5:
            ehr_g = main_h5.create_group('ehr')
            
            for temp_file, processed_stays in chunk_results:
                if not processed_stays:  # 빈 청크는 건너뛰기
                    continue
                    
                with h5py.File(temp_file, 'r') as temp_h5:
                    for stay_id in processed_stays:
                        # 데이터 복사
                        stay_id = str(stay_id)
                        temp_group = temp_h5['ehr'][stay_id]
                        stay_g = ehr_g.create_group(stay_id)
                        
                        for key in temp_group.keys():
                            temp_group.copy(key, stay_g)
                
                # 임시 파일 삭제
                temp_file.unlink()
        
        # 임시 디렉토리 삭제
        temp_dir.rmdir()
    
    def _log_memory_usage(self, stage: str):
        """메모리 사용량 로깅"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append((stage, memory_mb))
        logger.info(f"Memory usage at {stage}: {memory_mb:.1f} MB")
    
    def _print_performance_summary(self):
        """성능 요약 출력"""
        logger.info("=== Performance Summary ===")
        
        # 전체 시간 계산
        if hasattr(self, 'start_time') and self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        
        logger.info("Memory Usage by Stage:")
        for stage, memory_mb in self.memory_usage:
            logger.info(f"  {stage}: {memory_mb:.1f} MB")
        
        if self.memory_usage:
            max_memory = max(memory_mb for _, memory_mb in self.memory_usage)
            min_memory = min(memory_mb for _, memory_mb in self.memory_usage)
            avg_memory = sum(memory_mb for _, memory_mb in self.memory_usage) / len(self.memory_usage)
            logger.info(f"  Peak memory usage: {max_memory:.1f} MB")
            logger.info(f"  Min memory usage: {min_memory:.1f} MB")
            logger.info(f"  Avg memory usage: {avg_memory:.1f} MB")
        
        logger.info("==========================")
    
    def _ensure_datetime(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """컬럼이 datetime 타입인지 확인하고 변환"""
        if column not in df.columns:
            return df
        
        # 데이터 품질 분석 (디버깅용)
        if hasattr(self.cfg, 'debug_datetime') and self.cfg.debug_datetime:
            self._analyze_datetime_column(df, column)
        
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            try:
                # 변환 전 샘플 데이터 로깅 (각 컬럼당 한 번만)
                if column not in self._logged_columns:
                    sample_values = df[column].dropna().head(10).tolist()
                    logger.info(f"Sample values in column '{column}' before conversion: {sample_values}")
                    self._logged_columns.add(column)
                
                # 첫 번째 시도: 기본 변환
                df[column] = pd.to_datetime(df[column], errors='coerce')
                
                # 변환 실패한 행 수 확인
                null_count = df[column].isna().sum()
                
                if null_count > 0:
                    logger.warning(f"Failed to convert {null_count} rows in column '{column}' to datetime")
                    
                    # 변환 실패한 값들의 샘플 확인
                    failed_mask = df[column].isna()
                    if failed_mask.any():
                        failed_values = df.loc[failed_mask, column].dropna().head(5).tolist()
                        logger.warning(f"Sample failed values in column '{column}': {failed_values}")
                    
                    # 변환 성공률 계산
                    total_rows = len(df)
                    success_rate = ((total_rows - null_count) / total_rows) * 100
                    logger.info(f"DateTime conversion success rate for '{column}': {success_rate:.2f}%")
                    
                    # 성공률이 낮으면 추가 처리 시도
                    if success_rate < 90:  # 90% 미만 성공 시
                        logger.info(f"Low conversion success rate ({success_rate:.2f}%). Attempting alternative conversion methods...")
                        df = self._try_alternative_datetime_conversion(df, column)
                    
            except Exception as e:
                logger.error(f"Error converting column '{column}' to datetime: {e}")
                raise
        
        return df
    
    def _try_alternative_datetime_conversion(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """대체 datetime 변환 방법 시도"""
        # 원본 데이터 백업
        original_values = df[column].copy()
        
        # 방법 1: 다양한 날짜 형식 시도
        date_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y',
            '%Y%m%d %H:%M:%S',
            '%Y%m%d'
        ]
        
        for date_format in date_formats:
            try:
                # 변환 실패한 행들만 대상으로 시도
                failed_mask = df[column].isna()
                if not failed_mask.any():
                    break
                    
                failed_values = original_values[failed_mask]
                converted = pd.to_datetime(failed_values, format=date_format, errors='coerce')
                
                # 성공한 변환 결과 적용
                success_mask = ~converted.isna()
                if success_mask.any():
                    df.loc[failed_mask & success_mask, column] = converted[success_mask]
                    logger.info(f"Successfully converted {success_mask.sum()} rows using format '{date_format}'")
                    
            except Exception as e:
                logger.debug(f"Format '{date_format}' failed: {e}")
                continue
        
        # 방법 2: Unix timestamp 변환 시도
        try:
            failed_mask = df[column].isna()
            if failed_mask.any():
                failed_values = original_values[failed_mask]
                # 숫자로 변환 가능한지 확인
                numeric_values = pd.to_numeric(failed_values, errors='coerce')
                valid_numeric = ~numeric_values.isna()
                
                if valid_numeric.any():
                    # Unix timestamp로 변환 (초 단위)
                    converted = pd.to_datetime(numeric_values[valid_numeric], unit='s', errors='coerce')
                    success_mask = ~converted.isna()
                    
                    if success_mask.any():
                        df.loc[failed_mask & valid_numeric & success_mask, column] = converted[success_mask]
                        logger.info(f"Successfully converted {success_mask.sum()} rows as Unix timestamp")
                        
        except Exception as e:
            logger.debug(f"Unix timestamp conversion failed: {e}")
        
        # 최종 성공률 확인
        final_null_count = df[column].isna().sum()
        total_rows = len(df)
        final_success_rate = ((total_rows - final_null_count) / total_rows) * 100
        logger.info(f"Final datetime conversion success rate for '{column}': {final_success_rate:.2f}%")
        
        return df
    
    def _analyze_datetime_column(self, df: pd.DataFrame, column: str) -> None:
        """datetime 컬럼의 데이터 품질 분석"""
        if column not in df.columns:
            return
        
        logger.info(f"=== Analyzing datetime column: {column} ===")
        
        # 기본 통계
        total_rows = len(df)
        null_count = df[column].isna().sum()
        non_null_count = total_rows - null_count
        
        logger.info(f"Total rows: {total_rows}")
        logger.info(f"Null values: {null_count} ({null_count/total_rows*100:.2f}%)")
        logger.info(f"Non-null values: {non_null_count} ({non_null_count/total_rows*100:.2f}%)")
        
        if non_null_count > 0:
            # 데이터 타입 분포
            value_types = df[column].dropna().apply(type).value_counts()
            logger.info(f"Value types: {value_types.to_dict()}")
            
            # 문자열 값들의 길이 분포
            if df[column].dtype == 'object':
                str_lengths = df[column].dropna().astype(str).str.len()
                logger.info(f"String length stats: min={str_lengths.min()}, max={str_lengths.max()}, mean={str_lengths.mean():.2f}")
                
                # 고유 값 수
                unique_count = df[column].nunique()
                logger.info(f"Unique values: {unique_count}")
                
                # 가장 빈번한 값들
                most_common = df[column].value_counts().head(10)
                logger.info(f"Most common values: {most_common.to_dict()}")
        
        logger.info("=" * 50)
    
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
        """Create training dataset with parallel processing."""
        idx = {'train': self.train_idx, 'valid': self.valid_idx}[split]
        if self.cfg.mode.debugging_mode:
            idx = idx[:10]
        
        # 데이터를 청크로 나누기
        chunks = [idx[i:i + self.chunk_size] for i in range(0, len(idx), self.chunk_size)]
        
        def process_chunk(chunk_idx, chunk_data):
            """청크 단위로 데이터셋 처리"""
            hf_dataset = defaultdict(list)
            unique_tokens_chunk = torch.tensor([])
            
            with h5py.File(self.dest_path / f"{self.cfg.data_name}.h5", 'r') as h5_data:
                for icu_id in chunk_data:
                    ehr = h5_data['ehr'][icu_id]
                    tokens, types = self._process_icu_events(ehr, split)
                    if tokens is not None:
                        hf_dataset['icu_id'].append(icu_id)
                        hf_dataset['tokens'].append(tokens.to(torch.int32))
                        hf_dataset['types'].append(types.to(torch.int32))
                        unique_tokens_chunk = torch.cat((unique_tokens_chunk, tokens.to(torch.int32))).unique()
            
            return hf_dataset, unique_tokens_chunk
        
        # 병렬로 청크 처리
        logger.info(f"Processing {len(chunks)} chunks for {split} dataset with {self.n_workers} workers")
        chunk_results = []
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(process_chunk, i, chunk) for i, chunk in enumerate(chunks)]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split} chunks"):
                chunk_results.append(future.result())
        
        # 결과 병합
        merged_dataset = defaultdict(list)
        for hf_dataset, unique_tokens_chunk in chunk_results:
            for key in hf_dataset:
                merged_dataset[key].extend(hf_dataset[key])
            with self._lock:
                self.unique_tokens = torch.cat((self.unique_tokens, unique_tokens_chunk)).unique()
        
        # 빈 데이터셋 체크
        if not merged_dataset or all(len(v) == 0 for v in merged_dataset.values()):
            logger.warning(f"No data found for {split} training dataset. Skipping save.")
            return
        
        logger.info(f"Creating {split} training dataset with {len(merged_dataset.get('icu_id', []))} examples")
        dataset = Dataset.from_dict(merged_dataset)
        dataset.save_to_disk(self.dest_path / f"{split}_dataset_{self.cfg.max_seq_len}")

    def _make_inference_dataset(self, split: str):
        """Create inference dataset with parallel processing."""
        idx = {'test': self.test_idx, 'valid': self.valid_idx}[split]
        if self.cfg.mode.debugging_mode:
            idx = idx[:50]
        
        # 데이터를 청크로 나누기
        chunks = [idx[i:i + self.chunk_size] for i in range(0, len(idx), self.chunk_size)]
        
        def process_chunk(chunk_idx, chunk_data):
            """청크 단위로 추론 데이터셋 처리"""
            hf_dataset = defaultdict(list)
            unique_tokens_chunk = torch.tensor([])
            
            with h5py.File(self.dest_path / f"{self.cfg.data_name}.h5", 'r') as h5_data:
                for icu_id in chunk_data:
                    ehr = h5_data['ehr'][icu_id]
                    events = self._process_icu_events_for_inference(ehr, split)
                    # breakpoint()
                    
                    for event_data in events:
                        hf_dataset['icu_id'].append(icu_id)
                        for k in event_data.keys():
                            if type(event_data[k])==float:
                                hf_dataset[k].append(event_data[k])
                            else:
                                hf_dataset[k].append(event_data[k].to(torch.int32))
                        if split == "test":
                            unique_tokens_chunk = torch.cat((unique_tokens_chunk, event_data['prompt_tokens'].to(torch.int32))).unique()
                            unique_tokens_chunk = torch.cat((unique_tokens_chunk, event_data['lab_tokens'].to(torch.int32))).unique()
            breakpoint()
            return hf_dataset, unique_tokens_chunk
        
        # 병렬로 청크 처리
        logger.info(f"Processing {len(chunks)} chunks for {split} inference dataset with {self.n_workers} workers")
        chunk_results = []
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(process_chunk, i, chunk) for i, chunk in enumerate(chunks)]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split} inference chunks"):
                chunk_results.append(future.result())
        
        # 결과 병합
        merged_dataset = defaultdict(list)
        for hf_dataset, unique_tokens_chunk in chunk_results:
            for key in hf_dataset:
                if type(hf_dataset[key])==float: 
                    hf_dataset[key] = [hf_dataset[key]]
                merged_dataset[key].extend(hf_dataset[key])
            with self._lock:
                self.unique_tokens = torch.cat((self.unique_tokens, unique_tokens_chunk)).unique()
        
        # 빈 데이터셋 체크
        if not merged_dataset or all(len(v) == 0 for v in merged_dataset.values()):
            logger.warning(f"No data found for {split} inference dataset. Skipping save.")
            return
        
        logger.info(f"Creating {split} inference dataset with {len(merged_dataset.get('icu_id', []))} examples")
        dataset = Dataset.from_dict(merged_dataset)
        dataset.save_to_disk(self.dest_path / f"{split}_prompt_dataset_{self.cfg.max_seq_len}")

    def _process_icu_events(self, ehr: h5py.Group, split: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Process ICU events for training."""
        base_info = ehr['base_info'][()].decode('utf-8') if self.cfg.data.base_info else ""
        base_tokens = torch.tensor(self.tokenizer.encode(base_info, add_special_tokens=False)) if base_info else torch.tensor([])
        base_types = torch.zeros_like(base_tokens)
        tokens, types = torch.tensor([]), torch.tensor([])
        prev_time = None
        
        # 새로운 HDF5 구조에서 데이터 읽기
        times = [t.decode('utf-8') if isinstance(t, bytes) else t for t in ehr['time'][:]]
        tablenames = [t.decode('utf-8') if isinstance(t, bytes) else t for t in ehr['tablename'][:]]
        itemnames = [t.decode('utf-8') if isinstance(t, bytes) else t for t in ehr['itemname'][:]]
        
        # datetime 정보 읽기 (시간 토큰 생성용)
        event_datetimes = []
        icu_intimes = []
        if 'event_datetime' in ehr and 'icu_intime' in ehr:
            event_datetimes = [pd.to_datetime(t.decode('utf-8') if isinstance(t, bytes) else t) for t in ehr['event_datetime'][:]]
            icu_intimes = [pd.to_datetime(t.decode('utf-8') if isinstance(t, bytes) else t) for t in ehr['icu_intime'][:]]
        
        # value 정보 읽기
        values = []
        if 'value' in ehr:
            values = [t.decode('utf-8') if isinstance(t, bytes) else t for t in ehr['value'][:]]
        
        # valueuom 정보 읽기
        valueuoms = []
        if 'valueuom' in ehr:
            valueuoms = [t.decode('utf-8') if isinstance(t, bytes) else t for t in ehr['valueuom'][:]]
        
        logger.debug(f"Processing stay with {len(times)} events")
        if len(times) > 0:
            table_counts = pd.Series(tablenames).value_counts()
            logger.debug(f"Table distribution: {table_counts.to_dict()}")
        
        for i, (time, tablename, itemname) in enumerate(zip(times, tablenames, itemnames)):
            # 시간 토큰 생성
            time_tokens = []
            if event_datetimes and icu_intimes and i < len(event_datetimes) and i < len(icu_intimes):
                time_tokens = self._generate_time_tokens(event_datetimes[i], icu_intimes[i])
            else:
                # fallback: 절대 시간 방식
                if self.cfg.data.timeoffset == 'abs':
                    time_gap = int(float(time)) - prev_time if prev_time is not None else int(float(time))
                    time_tokens = [f'[{time_gap}]']
                else:
                    time_tokens = [time]
            
            # 이벤트 텍스트 생성: [시간토큰들] + [tablename] + [itemname] + [value] + [valueuom]
            event_text = time_tokens + [tablename, itemname]
            
            # value 추가 (있는 경우)
            if values and i < len(values) and values[i] not in ('', 'nan', 'None'):
                event_text.append(values[i])
            
            # valueuom 추가 (있는 경우)  
            if valueuoms and i < len(valueuoms) and valueuoms[i] not in ('', 'nan', 'None'):
                event_text.append(valueuoms[i])
            
            if tablename in ['labevents', 'lab', 'observation_tables'] and itemname in ('_ _ _', '___'):
                continue
            event_toks, event_types = self._tokenize_event(event_text, split, prev_time, tablename)
            if len(event_toks)==0:
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
            prev_time = int(float(time)) if self.cfg.data.timeoffset == 'abs' else None
            
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
        
        # 새로운 HDF5 구조에서 데이터 읽기
        times = [t.decode('utf-8') if isinstance(t, bytes) else t for t in ehr['time'][:]]
        tablenames = [t.decode('utf-8') if isinstance(t, bytes) else t for t in ehr['tablename'][:]]
        itemnames = [t.decode('utf-8') if isinstance(t, bytes) else t for t in ehr['itemname'][:]]
        
        # datetime 정보 읽기 (시간 토큰 생성용)
        event_datetimes = []
        icu_intimes = []
        if 'event_datetime' in ehr and 'icu_intime' in ehr:
            event_datetimes = [pd.to_datetime(t.decode('utf-8') if isinstance(t, bytes) else t) for t in ehr['event_datetime'][:]]
            icu_intimes = [pd.to_datetime(t.decode('utf-8') if isinstance(t, bytes) else t) for t in ehr['icu_intime'][:]]
        
        # value 정보 읽기
        values = []
        if 'value' in ehr:
            values = [t.decode('utf-8') if isinstance(t, bytes) else t for t in ehr['value'][:]]
        
        # valueuom 정보 읽기
        valueuoms = []
        if 'valueuom' in ehr:
            valueuoms = [t.decode('utf-8') if isinstance(t, bytes) else t for t in ehr['valueuom'][:]]
        
        # logger.debug(f"Processing stay with {len(times)} events")
        if len(times) > 0:
            table_counts = pd.Series(tablenames).value_counts()
            # logger.debug(f"Table distribution: {table_counts.to_dict()}")
            lab_count = table_counts.get('labevents', 0)
            # logger.debug(f"Lab events count: {lab_count}")
        
        for i, (time, tablename, itemname) in enumerate(zip(times, tablenames, itemnames)):
            # 시간 토큰 생성
            time_tokens = []
            if event_datetimes and icu_intimes and i < len(event_datetimes) and i < len(icu_intimes):
                time_tokens = self._generate_time_tokens(event_datetimes[i], icu_intimes[i])
            else:
                # fallback: 절대 시간 방식
                if self.cfg.data.timeoffset == 'abs':
                    time_gap = int(float(time)) - prev_time if prev_time is not None else int(float(time))
                    time_tokens = [f'[{time_gap}]']
                else:
                    time_tokens = [time]
            
            # 이벤트 텍스트 생성: [시간토큰들] + [tablename] + [itemname] + [value] + [valueuom]
            event_text = time_tokens + [tablename, itemname]
            
            # value 추가 (있는 경우)
            current_value = None
            if values and i < len(values) and values[i] not in ('', 'nan', 'None', '_ _ _', '___'):
                event_text.append(values[i])
                try:
                    current_value = float(values[i])
                except (ValueError, TypeError):
                    current_value = None
            
            # valueuom 추가 (있는 경우)
            if valueuoms and i < len(valueuoms) and valueuoms[i] not in ('', 'nan', 'None', '_ _ _', '___'):
                event_text.append(valueuoms[i])
            
            if tablename in ['labevents', 'lab', 'observation_tables'] and itemname in ('_ _ _', '___'):
                continue
            event_toks, event_types = self._tokenize_event(event_text, split, prev_time, tablename)
            if len(event_toks)==0:
                continue
                
            if tablename in ['labevents', 'lab', 'observation_tables']:
                previous_value_tokens = item_recent_values.get(itemname, torch.tensor([]))
                prompt_tokens, prompt_types = self._remove_same_time_events(event_toks, icu_tokens, icu_types)
                length_fixed = len(base_tokens) + len(event_toks) if self.cfg.data.base_info else len(event_toks)
                if length_fixed + len(prompt_tokens) > self.cfg.max_seq_len:
                    prompt_tokens, prompt_types = self._cut_prompt(length_fixed, prompt_tokens, prompt_types)
                if len(prompt_tokens) > 0:
                    final_prompt = torch.cat((base_tokens, prompt_tokens)) if self.cfg.data.base_info else prompt_tokens
                    final_types = torch.cat((base_types, prompt_types)) if self.cfg.data.base_info else prompt_types
                    
                    # lab_value에 실제 값 사용 (있는 경우)
                    lab_value = current_value #if current_value is not None else float(time.replace(' ', ''))
                    
                    if lab_value is None:
                        logger.warning(f"Skipping non-numeric lab value for {itemname}: {values[i] if values and i < len(values) else 'N/A'}")
                        continue
                    
                    results.append({
                        'prompt_tokens': final_prompt.to(torch.int32),
                        'lab_tokens': event_toks.to(torch.int32),
                        'lab_types': event_types.to(torch.int32),
                        'lab_value': lab_value,
                        'previous_value_tokens': previous_value_tokens.to(torch.int32),
                        'mean_previous_values': item_mean_prev_values.get(itemname, [np.nan, np.nan])[1]
                    })
                
                # 평균 값 업데이트
                if current_value is not None:
                    if itemname not in item_mean_prev_values:
                        item_mean_prev_values[itemname] = (1, current_value)
                    else:
                        len_, mean_ = item_mean_prev_values[itemname]
                        new_mean = round((mean_ * len_ + current_value) / (len_ + 1), 2)
                        item_mean_prev_values[itemname] = (len_ + 1, new_mean)
                    item_recent_values[itemname] = torch.tensor(self.tokenizer.encode(' '.join(event_text), add_special_tokens=False))
                else:
                    logger.warning(f"Skipping non-numeric lab value for {itemname}: {values[i] if values and i < len(values) else 'N/A'}")
                    
            icu_tokens = torch.cat((icu_tokens, event_toks)).to(torch.int32)
            icu_types = torch.cat((icu_types, event_types)).to(torch.int32)
            prev_time = int(float(time)) if self.cfg.data.timeoffset == 'abs' else None
        
        logger.debug(f"Generated {len(results)} inference results")
        return results

    def _tokenize_event(self, event_text: List, split: str, prev_time: Optional[int], tablename: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize an event."""
        event_toks, event_types = [], []
        
        # 시간 토큰 개수 결정 (새로운 시간 토큰 형식인지 확인)
        time_token_count = 0
        for i, text in enumerate(event_text):
            if isinstance(text, str) and (
                text.startswith('[DAY_') or 
                text in ['[MON]', '[TUE]', '[WED]', '[THU]', '[FRI]', '[SAT]', '[SUN]'] or
                text.endswith('h]') or 
                text.endswith('m]')
            ):
                time_token_count = i + 1
            else:
                break
        
        # 최소 1개는 시간 토큰이어야 함
        if time_token_count == 0:
            time_token_count = 1
        
        table_text = event_text[time_token_count] if len(event_text) > time_token_count else ''
        item_text = event_text[time_token_count + 1] if len(event_text) > time_token_count + 1 else ''
        
        for j, c in enumerate(event_text):
            if c not in ('', 'nan', 'None'):
                current_text = str(c)
                
                # 숫자 버킷팅 처리 (value 부분에만 적용)
                if (self.cfg.data.num_bucket and 
                    j > time_token_count + 1 and  # tablename, itemname 이후의 값들에만 적용
                    self._is_convertible_to_float(current_text)):
                    current_text = self._num_bucket(split, table_text, item_text, current_text)
                    if current_text is None:
                        return torch.tensor([]), torch.tensor([])
                
                toks = self.tokenizer.encode(current_text, add_special_tokens=False)
                event_toks.extend(toks)
                
                # lab events의 item name 이후 부분을 lab type으로 분류
                type_t = 1 if (table_text in ['labevents', 'lab', 'observation_tables'] and j > time_token_count) else 0
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
        if table_name == 'labevents' and getattr(self.cfg, 'use_10_labitems', False):
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
        return icu[icu['los'] >= (self.cfg.data.min_los / 24)]

    def _extract_icu(self):
        """Extract HIRID data from tar files."""
        # Implement tar extraction as needed
        pass

    def _make_compatible(self, icu: pd.DataFrame) -> pd.DataFrame:
        icu = icu.rename(columns={'admissiontime': 'intime'})
        icu['intime'] = pd.to_datetime(icu['intime'], errors='coerce')
        
        dfs = []
        for table_name, table_info in self.ehr_info.table_candidates.items():
            df = pd.read_parquet(self.data_path / table_info['fname'])
            df = df[[table_info['timestamp'], self.stayid_key]]
            df[table_info['timestamp']] = pd.to_datetime(df[table_info['timestamp']], errors='coerce')
            dfs.append(df.rename(columns={table_info['timestamp']: 'timestamp'}))
        
        merged_df = pd.concat(dfs, ignore_index=True).groupby(self.stayid_key).agg(max_time=('timestamp', 'max')).reset_index()
        icu = icu.merge(merged_df, on=self.stayid_key, how='inner')
        
        # datetime 타입 확인 및 안전한 계산
        icu = self._ensure_datetime(icu, 'max_time')
        icu = self._ensure_datetime(icu, 'intime')
        
        # NaN 값 제거
        icu = icu.dropna(subset=['max_time', 'intime'])
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