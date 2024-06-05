from MRICTTranslator import logger
from MRICTTranslator.config.configuration import ConfigurationManager
from MRICTTranslator.components.data_ingestion import DataIngestion

STAGE_NAME = "Data Ingestion"


class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        ingestion_config = config.get_data_ingestion_config()
        obj = DataIngestion(config=ingestion_config)
        obj.download_file()
        obj.extract_zip_file()


if __name__ == "__main__":
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
