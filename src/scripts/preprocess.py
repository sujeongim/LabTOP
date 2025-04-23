import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Type
from core.utils.ehr_processor import EHRProcessorFactory, EHRBase
from core.utils.feature import MIMICIV, eICU, HIRID
import logging


logger = logging.getLogger(__name__)

class EHRFactory:
    """Factory for creating EHR instances."""
    _ehr_types: Dict[str, Type[EHRBase]] = {
        'mimiciv': MIMICIV,
        'eicu': eICU,
        'hirid': HIRID
    }

    @classmethod
    def create(cls, data_name: str, cfg: DictConfig) -> EHRBase:
        ehr_class = cls._ehr_types.get(data_name.lower())
        if not ehr_class:
            raise ValueError(f"Unsupported EHR data source: {data_name}")
        return ehr_class(cfg)

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for EHR data processing pipeline."""
    logger.info(OmegaConf.to_yaml(cfg))
    try:
        ehr_info = EHRFactory.create(cfg.data_name, cfg)
        processor = EHRProcessorFactory.create(cfg.data_name, cfg, ehr_info)
        processor.extract_data()
        processor.preprocess()
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        raise

if __name__ == "__main__":
    main()