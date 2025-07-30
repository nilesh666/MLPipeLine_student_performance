import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import dill

def save_object(file_path, ob):
    try:
        with open(file_path, "wb") as f:
            dill.dump(ob, f)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluation_metrics(true, predicted):
  mae = mean_absolute_error(true, predicted)
  mse = mean_squared_error(true, predicted)
  rmse = np.sqrt(mse)
  r2_square = r2_score(true, predicted)
  return mae, rmse, r2_square

def evaluate_models(x_train, y_train, x_test, y_test, models):
    model_list = []
    results = []
    best_model = None
    best_r2 = float('-inf')  # to track best R² score

    for i in range(len(models)):
        model_name = list(models.keys())[i]
        model = list(models.values())[i]

        model.fit(x_train, y_train)

        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        model_train_mae, model_train_rmse, model_train_r2 = evaluation_metrics(y_train, y_train_pred)
        model_test_mae, model_test_rmse, model_test_r2 = evaluation_metrics(y_test, y_test_pred)

        results.append({
            "Model": model_name,
            "MAE_train": model_train_mae,
            "RMSE_train": model_train_rmse,
            "R2_train": model_train_r2,
            "MAE_test": model_test_mae,
            "RMSE_test": model_test_rmse,
            "R2_test": model_test_r2
        })

        # Track best model based on test R²
        if model_test_r2 > best_r2:
            best_r2 = model_test_r2
            best_model = model

    result_data = pd.DataFrame(results)
    result_data.to_csv("./artifacts/model_results.csv", index=False)

    return result_data, best_model

def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return dill.load(f)
        
    except Exception as e:
        raise CustomException(e, sys)