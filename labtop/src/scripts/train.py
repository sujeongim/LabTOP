import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig

from src.core.models.model import LabTOPModel
from src.core.models.trainer import Trainer
from src.core.data.dataloader import get_dataloader
from src.core.utils.helpers import ensure_dir, get_tokenizer, make_dataset
from torch.utils.data import Subset


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Initialize model
    model = LabTOPModel(cfg)

    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    tokenizer = get_tokenizer(cfg)
    train_dataset, valid_dataset, valid_prompt_dataset, _ = make_dataset(cfg, tokenizer, prompt_test=False)

    # Initialize trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(cfg, model, tokenizer, train_dataset, valid_dataset, valid_prompt_dataset, device)

    # Load data
    train_loader = get_dataloader(cfg, split="train")

    # Ensure checkpoint directory
    save_path = f"experiments/checkpoints/exp_{cfg.experiment_id}"
    ensure_dir(save_path)

    # Train model
    trainer.train(train_loader, save_path)


if __name__ == "__main__":
    main()