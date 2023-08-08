import os
import sys
import numpy as np
from dataclasses import dataclass

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            n_samples_max = 200
            X_train,y_train,X_test,y_test=(
                train_array[0:n_samples_max],
                train_array[0:n_samples_max],
                test_array[0:n_samples_max],
                test_array[0:n_samples_max]
            )
            models = {
                "Logistic Regression": LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                           class_weight='balanced', random_state=None, solver='lbfgs', max_iter=100, 
                           multi_class='ovr', verbose=0, warm_start=False, n_jobs=None),
            }
            params={

                "Logistic Regression":{}
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test) 
            accuracy_before_attack = accuracy_score(y_test, predicted)
            return accuracy_before_attack
        except Exception as e:
            raise CustomException(e,sys)

