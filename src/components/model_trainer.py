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

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        train_array = np.array(train_array)
        test_array = np.array(test_array)
        try:
            logging.info("Split training and test input data")
            n_samples_max = 200
            X_train, y_train, X_test, y_test = (
                train_array[0:n_samples_max],
                train_array[0:n_samples_max],
                test_array[0:n_samples_max],
                test_array[0:n_samples_max]
            )
            model = LogisticRegression(
                penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                intercept_scaling=1, class_weight='balanced', random_state=None,
                solver='lbfgs', max_iter=100, multi_class='ovr', verbose=0,
                warm_start=False, n_jobs=None
            )

            model.fit(X_train, y_train)

            # Generate adversarial examples after training the model
            art_classifier = SklearnClassifier(model)
            pgd = ProjectedGradientDescent(
                estimator=art_classifier, norm=np.inf, eps=.3, eps_step=0.1,
                max_iter=20, targeted=False, num_random_init=0, batch_size=100
            )

            x_train_adv = pgd.generate(X_train)
            x_test_adv = pgd.generate(X_test)

            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            model_scores = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                x_test_adv=x_test_adv, model=model
            )

            accuracy_before_attack = model.score(X_test, y_test)
            accuracy_after_attack = model.score(x_test_adv, y_test)
            return accuracy_before_attack, accuracy_after_attack

        except Exception as e:
            raise CustomException(e, sys)
