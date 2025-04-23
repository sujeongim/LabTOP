import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from torch.nn.utils.rnn import pad_sequence



class EHRGPTDataLoader:
    """DataLoader for EHRGPTDataset, handling batch collation and padding."""
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True
    ):
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn
        )

    def collate_fn(
        self, 
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collates a batch of data, padding sequences as needed."""
        input_ids = []
        train_masks = []
        pos_encs = []

        for input_data, train_mask, pos_enc in batch:
            input_ids.append(input_data)
            train_masks.append(train_mask)
            pos_encs.append(pos_enc)

        # Pad sequences to the maximum length in the batch
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        train_masks = pad_sequence(train_masks, batch_first=True, padding_value=0)
        pos_encs = pad_sequence(pos_encs, batch_first=True, padding_value=0)

        return input_ids, train_masks, pos_encs

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


class PromptTestDataLoader:
    """DataLoader for PromptTestDataset, handling batch collation and padding."""
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True
    ):
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn
        )

    def collate_fn(
        self, 
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collates a batch of test data, padding sequences as needed."""
        prompts = []
        label_values = []
        previous_values = []
        mean_previous_values = []

        for prompt, label_value, prev_value, mean_prev_values in batch:
            prompts.append(prompt)
            label_values.append(label_value)
            previous_values.append(prev_value)
            mean_previous_values.append(mean_prev_values)

        # Pad sequences to the maximum length in the batch
        prompts = pad_sequence(prompts, batch_first=True, padding_value=0)
        previous_values = pad_sequence(previous_values, batch_first=True, padding_value=0)
        
        # Stack non-sequence tensors
        label_values = torch.stack([lv if lv.dim() > 0 else lv.unsqueeze(0) for lv in label_values])
        mean_previous_values = torch.stack(mean_previous_values)

        return prompts, label_values, previous_values, mean_previous_values

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)
    
    
def get_dataloader(
    dataset: Union[EHRGPTDataset, PromptTestDataset],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Union[EHRGPTDataLoader, PromptTestDataLoader]:
    """
    Creates and returns a DataLoader for the given dataset.
    
    Args:
        dataset: Either EHRGPTDataset or PromptTestDataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data (default: True for training, False for testing)
        num_workers: Number of subprocesses for data loading
        pin_memory: If True, copies tensors into CUDA pinned memory (recommended for GPU training)
    
    Returns:
        An instance of EHRGPTDataLoader or PromptTestDataLoader
    """
    if isinstance(dataset, EHRGPTDataset):
        return EHRGPTDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    elif isinstance(dataset, PromptTestDataset):
        return PromptTestDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")