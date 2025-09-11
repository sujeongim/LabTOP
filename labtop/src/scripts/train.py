import sys
import os
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from core.models.model import LabTOPModel
from core.models.trainer import Trainer
from core.data.dataloader import get_dataloader
from core.utils.helpers import ensure_dir, get_tokenizer, make_dataset
from torch.utils.data import Subset

from core.utils.logging import get_logger
logger = get_logger("train")  

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logger.info("Tokenizer and Dataset Initialization")
    tokenizer = get_tokenizer(cfg)
    train_dataset, valid_dataset, valid_prompt_dataset, _ = make_dataset(cfg, tokenizer, prompt_test=False)

    # Initialize model
    logger.info("Initializing LabTOP Model")
    model = LabTOPModel(cfg, tokenizer)

    logger.info("Optimizer and Loss Initialization")
    # Initialize optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize trainer
    logger.info("Initializing Trainer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(cfg, model, tokenizer, train_dataset, valid_dataset, valid_prompt_dataset, optimizer, criterion, device)

    # Ensure checkpoint directory
    #save_path = f"experiments/checkpoints/exp_{cfg.experiment_id}"
    #ensure_dir(save_path)

    # Train model
    trainer.train()


if __name__ == "__main__":
    main()