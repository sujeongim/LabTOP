import os
from typing import Tuple, List, Optional
import h5py
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset
from transformers import AutoTokenizer


class TimeBasedPositionalEncoder:
    """Handles time-based positional encoding for EHR data."""
    
    def __init__(self, tokenizer: AutoTokenizer, cfg):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.day_tokens = torch.tensor(
            tokenizer.encode(
                ' '.join(f'[DAY_{i}]' for i in range(1, cfg.data.max_day_len + 1)),
                add_special_tokens=False
            )
        )
        self.time_tok_num = 4 if cfg.data.add_weekday else 3

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Creates positional encoding based on time boundaries."""
        time_idx_list = torch.where(torch.isin(input_ids, self.day_tokens))[0]
        base_info = input_ids[:time_idx_list[0]]
        
        pos_enc_by_time = [torch.tensor([0] * len(base_info))]
        pos_id = 1
        pos_start = time_idx_list[0]

        for i in range(len(time_idx_list) - 1):
            time_s, time_e = time_idx_list[i], time_idx_list[i + 1]
            if not torch.equal(
                input_ids[time_s:time_s + self.time_tok_num],
                input_ids[time_e:time_e + self.time_tok_num]
            ):
                pos_enc_by_time.append(torch.tensor([pos_id] * (time_e - pos_start)))
                pos_id += 1
                pos_start = time_e

        pos_enc_by_time.append(torch.tensor([pos_id] * (len(input_ids) - pos_start)))
        return torch.cat(pos_enc_by_time)


class EventPermuter:
    """Handles random permutation of same-time events."""
    
    def __init__(self, day_tokens: torch.Tensor, time_tok_num: int):
        self.day_tokens = day_tokens
        self.time_tok_num = time_tok_num

    def permute_events(
        self, 
        same_time_events: List[torch.Tensor], 
        same_train_mask: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Permutes events and their corresponding masks."""
        permuted_idx = torch.randperm(len(same_time_events))
        return (
            [same_time_events[i] for i in permuted_idx],
            [same_train_mask[i] for i in permuted_idx]
        )

    def randomize_events(
        self, 
        input_ids: torch.Tensor, 
        train_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly permutes events within same time periods."""
        time_idx_list = torch.where(torch.isin(input_ids, self.day_tokens))[0]
        time_idx_list = torch.cat((time_idx_list, torch.tensor([len(input_ids)])))
        
        base_info = input_ids[:time_idx_list[0]]
        base_info_mask = train_mask[:time_idx_list[0]]
        
        total_events = [base_info]
        total_train_mask = [base_info_mask]
        same_time_events = []
        same_train_mask = []

        for i in range(len(time_idx_list) - 1):
            time_s, time_e = time_idx_list[i], time_idx_list[i + 1]
            event = input_ids[time_s:time_e]
            train = train_mask[time_s:time_e]
            
            if not same_time_events or torch.equal(
                same_time_events[-1][:self.time_tok_num], 
                event[:self.time_tok_num]
            ):
                same_time_events.append(event)
                same_train_mask.append(train)
            else:
                permuted_events, permuted_mask = self.permute_events(
                    same_time_events, same_train_mask
                )
                total_events += permuted_events
                total_train_mask += permuted_mask
                same_time_events = [event]
                same_train_mask = [train]

        permuted_events, permuted_mask = self.permute_events(same_time_events, same_train_mask)
        total_events += permuted_events
        total_train_mask += permuted_mask
        
        return torch.cat(total_events), torch.cat(total_train_mask)


class EHRGPTDataset:
    """Dataset class for EHR GPT model training."""
    
    def __init__(
        self,
        cfg,
        tokenizer: AutoTokenizer,
        data_path: str,
        train: bool = True,
    ):
        self.cfg = cfg
        self.dataset = Dataset.load_from_disk(data_path)
        self.tokenizer = tokenizer
        self.train = train
        self.train_only_lab = cfg.data.train_only_lab
        
        self.pos_encoder = TimeBasedPositionalEncoder(tokenizer, cfg)
        self.event_permuter = EventPermuter(
            self.pos_encoder.day_tokens,
            self.pos_encoder.time_tok_num
        )

    def __len__(self) -> int:
        """Returns the size of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves a single item from the dataset."""
        assert not (self.cfg.train.type_token or self.cfg.train.dpe), \
            "type_token and dpe are not yet implemented."

        input_data = torch.tensor(self.dataset[idx]['tokens'], dtype=torch.long)
        input_train_mask = torch.tensor(
            self.dataset[idx]['types'] if self.train_only_lab else torch.ones(input_data.shape),
            dtype=torch.long
        )

        if self.train and self.cfg.train.rand_perm and not self.cfg.data.no_time:
            input_data, input_train_mask = self.event_permuter.randomize_events(
                input_data, input_train_mask
            )
            
            assert input_data.shape == input_train_mask.shape
            assert torch.all(input_data.unique() == input_data.unique())
            assert torch.all(input_train_mask.unique() == input_train_mask.unique())

        pos_enc = (
            self.pos_encoder.encode(input_data)
            if self.cfg.train.pos_enc_by_time
            else torch.arange(len(input_data))
        )

        return input_data, input_train_mask, pos_enc.long()


class PromptTestDataset:
    """Dataset class for prompt-based testing of EHR GPT model."""
    
    def __init__(
        self,
        cfg,
        tokenizer: AutoTokenizer,
        prompt_data_path: str,
        prompt_test: bool = True
    ):
        self.cfg = cfg
        self.event_type = cfg.test.event_type
        self.tokenizer = tokenizer
        self.test_dataset = Dataset.load_from_disk(prompt_data_path)
        
        self.pos_encoder = TimeBasedPositionalEncoder(tokenizer, cfg)
        self.event_permuter = EventPermuter(
            self.pos_encoder.day_tokens,
            self.pos_encoder.time_tok_num
        )

    def __len__(self) -> int:
        """Returns the size of the dataset."""
        return len(self.test_dataset)

    def _cut_event(self, input_id: torch.Tensor) -> torch.Tensor:
        """Truncates events to fit within max sequence length."""
        time_token_poses = torch.where(torch.isin(input_id, self.pos_encoder.day_tokens))[0]
        i = 0
        curr_time_tok = time_token_poses[0]
        next_time_tok = time_token_poses[1]
        
        while len(input_id) > self.cfg.max_seq_len - 15:
            remove_len = len(input_id[curr_time_tok:next_time_tok])
            front = input_id[:curr_time_tok]
            back = input_id[next_time_tok:]
            input_id = torch.cat((front, back), dim=0)
            curr_time_tok = time_token_poses[i + 1] - remove_len
            next_time_tok = time_token_poses[i + 2] - remove_len
            i += 1
        return input_id

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves a single test item from the dataset."""
        test_data = self.test_dataset[index]
        if 'propmt_tokens' in test_data:
            test_data['prompt_tokens'] = test_data.pop('propmt_tokens')

        prompt = torch.tensor(test_data['prompt_tokens'], dtype=torch.long)
        label = torch.tensor(test_data['lab_tokens'], dtype=torch.long)
        label_type = torch.tensor(test_data['lab_types'], dtype=torch.long) if self.cfg.data.no_col_name else None
        previous_value = torch.tensor(test_data['previous_value_tokens'], dtype=torch.long)
        mean_previous_values = torch.tensor(test_data['mean_previous_values'], dtype=torch.float)

        if self.cfg.test.rand_perm and not self.cfg.data.no_time:
            prompt, _ = self.event_permuter.randomize_events(prompt, torch.ones_like(prompt))
            
        pos_enc = (
            self.pos_encoder.encode(prompt)
            if self.cfg.train.pos_enc_by_time
            else torch.arange(len(prompt))
        )

        # Process label and concatenate with prompt
        value_idx = torch.where(label_type == 1)[0][
            0 if self.cfg.data.add_end_of_event else self.pos_encoder.time_tok_num
        ].item()
        label_before_value = label[:value_idx]
        label_value = (
            torch.tensor(test_data['lab_value'], dtype=torch.float)
            if self.cfg.data.num_bucket
            else label[value_idx:]
        )

        prompt = torch.cat((prompt, label_before_value))
        return prompt, label_value, previous_value, mean_previous_values