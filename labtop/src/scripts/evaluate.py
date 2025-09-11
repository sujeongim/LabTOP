import yaml
import torch
import os
import sys
import hydra
from omegaconf import DictConfig
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from core.models.model import LabTOPModel
from core.models.inference import Inference
from core.utils.metrics import PerformanceEvaluator
from core.utils.logging import get_logger

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def evaluate(cfg:DictConfig):
    # Initialize inference
    inference = Inference(cfg)
    logger = get_logger("evaluate")
    
    save_path = inference.predict()
        
    # Compute metrics
    evaluator = PerformanceEvaluator()
    weighted_nmae, weighted_smape = evaluator.calculate(save_path)
   
    # Log results
    logger.info("Evaluation Results:")
    logger.info(f"Weighted NMAE: {weighted_nmae}")
    logger.info(f"Weighted SMAPE: {weighted_smape}")
    
if __name__ == "__main__":
    evaluate()