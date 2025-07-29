import sys
import os
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_ob(self):
        try:
            num_features = ["writing score", "reading score"]
            cat_features = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            num_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder()),
                ]
            )

            logging.info(f"Numerical columns are {num_features}")
            logging.info(f"Categorical columns are {cat_features}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_features),
                    ("cat_pipeline", cat_pipeline, cat_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_ob = self.get_data_transformer_ob()

            target_col = "math score"
            num_col = ["writing score", "reading score"]

            logging.info(f"Columns of train_df: {train_df.columns}")

            input_feature_train = train_df.drop(columns=[target_col], axis = 1)
            target_feature_train = train_df[target_col]
            
            input_feature_test = test_df.drop(columns=[target_col], axis=1)
            target_feature_test = test_df[target_col]

            logging.info("Applying preprocessing object on train and test data")
            logging.info(f"Columns of input_feature_train: {input_feature_train.columns}")

            input_feature_train_arr = preprocessor_ob.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessor_ob.transform(input_feature_test)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                ob = preprocessor_ob
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)