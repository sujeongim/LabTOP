import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Type
import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.utils.ehr_processor import EHRProcessorFactory, EHRBase
from core.utils.feature import MIMICIV, eICU, HIRID


logger = logging.getLogger(__name__)

class EHRConfigValidator:
    """Validates EHR configuration parameters."""
    
    @staticmethod
    def validate(cfg: DictConfig) -> None:
        """Validates required configuration parameters."""
        required_fields = ['data_name']
        for field in required_fields:
            if not hasattr(cfg, field):
                raise ValueError(f"Missing required configuration field: {field}")
        
        if cfg.data_name.lower() not in EHRFactory.get_supported_datasets():
            raise ValueError(f"Unsupported data source: {cfg.data_name}")

class EHRFactory:
    """Factory for creating EHR instances."""
    
    _ehr_types: dict[str, Type[EHRBase]] = {
        'mimiciv': MIMICIV,
        'eicu': eICU,
        'hirid': HIRID
    }

    @classmethod
    def get_supported_datasets(cls) -> set[str]:
        """Returns set of supported dataset names."""
        return set(cls._ehr_types.keys())

    @classmethod
    def create(cls, data_name: str, cfg: DictConfig) -> EHRBase:
        """Creates an EHR instance based on data source."""
        ehr_class = cls._ehr_types.get(data_name.lower())
        if not ehr_class:
            raise ValueError(f"Unsupported EHR data source: {data_name}")
        return ehr_class(cfg)

class EHRPipeline:
    """Manages the EHR data processing pipeline."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize pipeline with configuration."""
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

    def execute(self) -> None:
        """Executes the complete EHR processing pipeline."""
        try:
            self.logger.info("Starting EHR processing pipeline")
            self.logger.info(f"Configuration:\n{OmegaConf.to_yaml(self.cfg)}")
            
            # Validate configuration
            EHRConfigValidator.validate(self.cfg)
            
            # Create EHR instance and processor
            ehr_info = EHRFactory.create(self.cfg.data_name, self.cfg)
            processor = EHRProcessorFactory.create(self.cfg.data_name, self.cfg, ehr_info)
            
            # Execute pipeline steps
            processor.extract_data()
            processor.preprocess()
            
            self.logger.info("Pipeline completed successfully")
            
        except ValueError as ve:
            self.logger.error(f"Configuration error: {ve}")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            sys.exit(1)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for EHR data processing pipeline."""
    pipeline = EHRPipeline(cfg)
    pipeline.execute()

if __name__ == "__main__":
    main()