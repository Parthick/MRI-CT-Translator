from MRICTTranslator import logger
from MRICTTranslator.pipeline.stage_01_dta_ingestion import DataIngestionPipeline
from MRICTTranslator.pipeline.stage_02_model_trainer import ModelTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f"\\n The {STAGE_NAME} has started <<<<<<<<<<<< \n\n")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(
        f">>>>>>>>>>> The {STAGE_NAME} has completed successfully <<<<<<<<<< \n\n ==========="
    )
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training Stage"
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
