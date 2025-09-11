import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from omegaconf import OmegaConf, DictConfig
from datasets import Dataset
from torch.utils.data import Sampler, DataLoader
import hydra
import pickle
import pandas as pd
from transformers import AutoTokenizer
import argparse

RAW_DATA_PATH = "/home/data_storage/mimic-iv-3.1/"
PROCESSED_DATA_PATH = "/home/edlab/sjim/LabTOP/labtop/data/mimiciv/mimiciv_procedureevents_labevents_numeric_datetime"

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

def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    special_tokens = build_special_tokens(cfg)
    
    if special_tokens:
        num_added = tokenizer.add_special_tokens({
            "additional_special_tokens": special_tokens
        })
        print("We have added", num_added, "tokens\n")
    
    return tokenizer

def assign_stay_id_by_time(raw_data, icu_data, table_name):
    """
    이벤트 시간이 ICU 입원/퇴원 시간 사이에 있는 경우에만 해당 stay_id를 부여
    """
    print(f"Processing {table_name} - assigning stay_id based on time overlap...")
    
    # 테이블별 timestamp 컬럼명 매핑
    timestamp_columns = {
        'labevents': 'charttime',
        'procedureevents': 'starttime',
        'inputevents': 'starttime',
        'outputevents': 'charttime',
        'microbiologyevents': 'charttime'
    }
    
    timestamp_col = timestamp_columns.get(table_name)
    if not timestamp_col:
        print(f"Warning: Unknown timestamp column for table {table_name}, using charttime as default")
        timestamp_col = 'charttime'
    
    if timestamp_col not in raw_data.columns:
        print(f"Warning: {timestamp_col} not found in {table_name}, trying alternative columns...")
        # 대안 컬럼명들 시도
        alternative_cols = ['charttime', 'starttime', 'endtime', 'storetime']
        for alt_col in alternative_cols:
            if alt_col in raw_data.columns:
                timestamp_col = alt_col
                print(f"Using {alt_col} as timestamp column for {table_name}")
                break
        else:
            print(f"Error: No valid timestamp column found in {table_name}")
            return raw_data
    
    # datetime 변환
    print(f"Converting {timestamp_col} to datetime...")
    raw_data[timestamp_col] = pd.to_datetime(raw_data[timestamp_col], errors='coerce')
    icu_data['intime'] = pd.to_datetime(icu_data['intime'], errors='coerce')
    icu_data['outtime'] = pd.to_datetime(icu_data['outtime'], errors='coerce')
    
    # NaN 값 제거
    initial_count = len(raw_data)
    raw_data = raw_data.dropna(subset=[timestamp_col])
    after_time_filter = len(raw_data)
    print(f"Removed {initial_count - after_time_filter} rows with invalid timestamps")
    
    # hadm_id가 있는 icu_data만 사용
    valid_icu = icu_data.dropna(subset=['hadm_id', 'intime', 'outtime'])
    print(f"Valid ICU stays: {len(valid_icu)}")
    
    # 결과를 저장할 리스트
    results = []
    
    # hadm_id별로 그룹화하여 처리
    for hadm_id, hadm_group in raw_data.groupby('hadm_id'):
        if pd.isna(hadm_id):
            continue
            
        # 해당 hadm_id의 ICU stays 찾기
        icu_stays = valid_icu[valid_icu['hadm_id'] == hadm_id]
        
        if len(icu_stays) == 0:
            # ICU stay가 없는 경우 stay_id를 NaN으로 설정
            hadm_group = hadm_group.copy()
            hadm_group['stay_id'] = None
            hadm_group['subject_id'] = None
            results.append(hadm_group)
            continue
        
        # 각 이벤트에 대해 적절한 stay_id 찾기
        for idx, event in hadm_group.iterrows():
            event_time = event[timestamp_col]
            assigned_stay = None
            
            # 각 ICU stay와 시간 겹침 확인
            for _, icu_stay in icu_stays.iterrows():
                if (event_time >= icu_stay['intime'] and 
                    event_time <= icu_stay['outtime']):
                    assigned_stay = icu_stay
                    break
            
            # 결과 추가
            event_copy = event.copy()
            if assigned_stay is not None:
                event_copy['stay_id'] = assigned_stay['stay_id']
                event_copy['subject_id'] = assigned_stay['subject_id']
            else:
                event_copy['stay_id'] = None
                event_copy['subject_id'] = None
            
            results.append(event_copy.to_frame().T)
    
    # 결과 합치기
    if results:
        result_df = pd.concat(results, ignore_index=True)
    else:
        result_df = raw_data.copy()
        result_df['stay_id'] = None
        result_df['subject_id'] = None
    
    # 통계 출력
    total_events = len(result_df)
    events_with_stay = len(result_df[result_df['stay_id'].notna()])
    events_without_stay = total_events - events_with_stay
    
    print(f"Results for {table_name}:")
    print(f"  Total events: {total_events}")
    print(f"  Events with stay_id: {events_with_stay} ({events_with_stay/total_events*100:.1f}%)")
    print(f"  Events without stay_id: {events_without_stay} ({events_without_stay/total_events*100:.1f}%)")
    
    # stay_id별 이벤트 수 상위 10개
    if events_with_stay > 0:
        stay_counts = result_df[result_df['stay_id'].notna()]['stay_id'].value_counts().head(10)
        print(f"  Top 10 stays by event count: {dict(stay_counts)}")
    
    return result_df

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # 토크나이저 로드
    tokenizer = get_tokenizer(cfg)
    
    table_names = PROCESSED_DATA_PATH.split('/')[-1].split('_')[1:-2]
    print(f"테이블 이름: {table_names}")
    table_dict = {}
    
    # icu data
    icu_data = pd.read_csv(RAW_DATA_PATH + "icu/icustays.csv")
    
    # pure raw data
    for table_name in table_names:
        try : 
            raw_data = pd.read_csv(RAW_DATA_PATH + f"icu/{table_name}.csv" , low_memory=False)
        except FileNotFoundError:
            if table_name == "labevents":
                raw_data = pd.read_csv(RAW_DATA_PATH + f"hosp/labevents.csv" , low_memory=False, nrows=50000)
            else:
                raw_data = pd.read_csv(RAW_DATA_PATH + f"hosp/{table_name}.csv" , low_memory=False)
            
            # 시간 기반으로 적절한 stay_id 부여
            raw_data = assign_stay_id_by_time(raw_data, icu_data, table_name)
            
            
        filtered_data_path = PROCESSED_DATA_PATH + f"/{table_name}_filtered.csv"
        table_dict[table_name] = {
                "raw": raw_data,
                "filtered": pd.read_csv(filtered_data_path, low_memory=False)
            }
    
    # 전처리된 데이터 로드
    preprocessed_data = Dataset.load_from_disk(PROCESSED_DATA_PATH + "/train_dataset_4096")
    
    if len(preprocessed_data) == 0:
        print("전처리된 데이터가 비어있습니다.")
        return
        
    
    for data in preprocessed_data.select(range(0, 5)):
        icu_stay_id = int(data.get('icu_id', None))
        tokens = data.get('tokens', None)
        
        if icu_stay_id is None:
            print("stay_id가 없습니다. 전처리된 데이터가 올바르지 않습니다.")
            print(data.keys())
            return
        print(f"ICU Stay ID: {icu_stay_id}")
        icu_stay_data = icu_data[icu_data['stay_id'] == icu_stay_id]
        print(f"ICU Stay Data: {icu_stay_data}")
        
        print(f"Preprocessed data:")
        print(tokenizer.decode(tokens))
        print()
        
        for table_name, table in table_dict.items():
            raw_data = table['raw']
            filtered_data = table['filtered']
            
            if icu_stay_id not in raw_data['stay_id'].values:
                print(f"Raw data에서 stay_id {icu_stay_id}를 찾을 수 없습니다. 테이블: {table_name}")
                continue
            else:
                raw_row = raw_data[raw_data['stay_id'] == icu_stay_id]
                if len(raw_row) == 0:
                    print(f"Raw data에서 stay_id {icu_stay_id}에 해당하는 행이 없습니다. 테이블: {table_name}")
                    continue

            if icu_stay_id not in filtered_data['stay_id'].values:
                print(f"Filtered data에서 stay_id {icu_stay_id}를 찾을 수 없습니다. 테이블: {table_name}")
                continue
            else:
                filtered_row = filtered_data[filtered_data['stay_id'] == icu_stay_id]
                if len(filtered_row) == 0:
                    print(f"Filtered data에서 stay_id {icu_stay_id}에 해당하는 행이 없습니다. 테이블: {table_name}")
                    continue

            
            print(f"테이블: {table_name}, stay_id: {icu_stay_id}")
            
            # 원본 데이터에서 해당 stay_id의 이벤트들을 시간순으로 정렬
            raw_events = raw_row.sort_values(by=[col for col in raw_row.columns if 'time' in col.lower()][:1] or ['hadm_id'])
            print(f"원본 데이터 이벤트 수: {len(raw_events)}")
            
            # 원본 데이터 샘플 출력 (시간순)
            if len(raw_events) > 0:
                print("원본 데이터 샘플 [시간순 정렬]:")
                timestamp_cols = [col for col in raw_events.columns if 'time' in col.lower()]
                if timestamp_cols:
                    time_col = timestamp_cols[0]
                    for i, (_, event) in enumerate(raw_events.head(15).iterrows(), 1):
                        item_id = event.get('itemid', 'N/A')
                        value = event.get('valuenum', event.get('value', 'N/A'))
                        print(f"  {i:2d}. {event[time_col]} | ItemID: {item_id} | Value: {value}")
                
                # ICU stay 시간 정보
                intime = icu_stay_data.iloc[0]['intime'] if len(icu_stay_data) > 0 else 'N/A'
                outtime = icu_stay_data.iloc[0]['outtime'] if len(icu_stay_data) > 0 else 'N/A'
                print(f"  - 시작 시간: {intime}")
                print(f"  - 종료 시간: {outtime}")
            
            # 필터링된 데이터 샘플 출력
            print(f"필터링된 데이터 이벤트 수: {len(filtered_row)}")
            if len(filtered_row) > 0:
                print("필터링된 데이터 샘플:")
                for i, (_, event) in enumerate(filtered_row.head(10).iterrows(), 1):
                    item_name = event.get('itemname_text', 'N/A')
                    value = event.get('value', 'N/A')
                    time_val = event.get('time', 'N/A')
                    print(f"  {i:2d}. {time_val} | ItemName: {item_name} | Value: {value}")
            
            print("-" * 40)
    print("-" * 60)

        

    

if __name__ == "__main__":
    main()