import yaml
import torch
from src.core.models.model import LabTOPModel
from src.core.models.inference import Inference
from src.core.utils.metrics import PerformanceEvaluator
from src.core.utils.logging import get_logger

def evaluate():
    # Load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize inference
    inference = Inference(config)
    logger = get_logger("evaluate")
    
    save_path = inference.predict()
        
    # Compute metrics
    evaluator = PerformanceEvaluator()
    weighted_nmae, weighted_smape = evaluator.calculate(save_path)
    print("Evaluation Results:")
    print(f"Weighted NMAE: {weighted_nmae}")
    print(f"Weighted SMAPE: {weighted_smape}")

if __name__ == "__main__":
    evaluate()