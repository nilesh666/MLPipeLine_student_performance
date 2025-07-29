import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill

def save_object(file_path, ob):
    try:
        with open(file_path, "wb") as f:
            dill.dump(ob, f)

    except Exception as e:
        raise CustomException(e, sys)