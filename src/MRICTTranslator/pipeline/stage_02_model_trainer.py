from MRICTTranslator import logger
from MRICTTranslator.config.configuration import ConfigurationManager
from MRICTTranslator.components.model_trainer import Training

STAGE_NAME = "Model Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_training_config()
        model_train = Training(config=model_training_config)
        model_train.model_train()


if __name__ == "__main__":
    try:
        logger.info(f"\\n The {STAGE_NAME} has started <<<<<<<<<<<< \n\n")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>>>>>>> The {STAGE_NAME} has completed successfully <<<<<<<<<< \n\n ==========="
        )
    except Exception as e:
        logger.exception(e)
        raise e
