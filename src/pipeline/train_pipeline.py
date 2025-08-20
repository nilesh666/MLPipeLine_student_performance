from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import sys


if __name__ =="__main__":

    try:
        logging.info("Training Pipeline initiated")

        data_ingest = DataIngestion()
        train_path, test_path = data_ingest.initiate_data_ingestion()

        transform_data = DataTransformation()
        train_arr, test_arr , _= transform_data.initiate_data_transformation(train_path, test_path)

        model = ModelTrainer()
        model.initiate_model_trainer(train_arr, test_arr)

        logging.info("Finished training pipeline")

    except Exception as e:
        logging.error("Erro in training pipeline")
        raise CustomException(e, sys)

