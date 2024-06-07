from MRICTTranslator.constants import *
from MRICTTranslator.utils.common import read_yaml, create_directories
from MRICTTranslator.entity.config_entity import (
    DataIngestionConfig,
    ModelTrainingConfig,
)


class ConfigurationManager:
    def __init__(
        self, config_file_path=CONFIG_FILE_PATH, params_file_path=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )
        return data_ingestion_config

    def get_training_config(self) -> ModelTrainingConfig:
        config = self.config.training
        create_directories([config.root_dir])
        params = self.params
        dataset = "artifacts/data_ingestion/dataset/Dataset/images/"
        training_config = ModelTrainingConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            dataset=str(dataset),
            img_shape=params.IMAGE_SHAPE,
            epochs=params.EPOCHS,
            lr=params.LR,
            l1=params.L1,
            l2=params.L2,
        )
        return training_config
