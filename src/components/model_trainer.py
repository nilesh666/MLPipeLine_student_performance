import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('./artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try: 
            logging.info("Input the data and split")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }


            model_report, best_model= evaluate_models(x_train = x_train, 
                                                y_train = y_train,
                                                x_test = x_test,
                                                y_test = y_test,
                                                models = models,
                                                )
            
            best_model_1 = model_report[model_report['R2_test'] == model_report['R2_test'].max()]
            best_model_name = best_model_1['Model'].iloc[0]
            best_model_score = best_model_1['R2_test'].iloc[0]

            if best_model_score < 0.6:
                raise CustomException("Model Poor")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_path,
                ob = best_model
            )
            
            logging.info(f"Models trained and the best model is {best_model_name} and the score is {best_model_score}!!!")


        except Exception as e:
            raise CustomException(e, sys)