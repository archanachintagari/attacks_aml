import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import accuracy_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, x_test_adv, model):
    try:
        report = {}

        for model_name, model in model.items():

            model.fit(X_train, y_train)  # Train model

            test_prediction1 = model.predict(X_test)
            test_prediction2 = model.predict(x_test_adv)

            train_model_score = accuracy_score(y_train, test_prediction1)

            test_model_score = accuracy_score(y_test, test_prediction2)

            # report[list(model.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)