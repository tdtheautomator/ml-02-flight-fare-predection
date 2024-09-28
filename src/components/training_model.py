#file used to create code for training model
import os
import sys

from dataclasses import dataclass
from src.tools.custom_exception import CustomException
from src.tools.custom_logger import logging
from src.tools.common import save_object, evaluate_model_best_param_gsv, evaluate_model_best_param_rsv


from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


@dataclass

@dataclass
class TrainingModelConfig:
    trained_model_file_path=os.path.join("outputs","trained_model.pkl")

class TrainingModel:
    def __init__(self):
        self.training_model_config=TrainingModelConfig()

    def initiate_training_model(self,training_array,test_array):
        logging.info("initiated training model")
        try:
            logging.info("spliting training and test data")

            X_training,y_training,X_test,y_test=(
                training_array[:,:-1], #all rows and columns except column
                training_array[:,-1],  #only last column
                test_array[:,:-1],     #all rows and columns except column
                test_array[:,-1]       #only last column
            )
            models = {
                "Linear Regression": LinearRegression(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "Random Forest": RandomForestRegressor()
            }
            #used for hyper tuning
            params={
                "Linear Regression":{},
                "CatBoosting Regressor":{},
                "Random Forest":{}
            }
            
            model_report:dict=evaluate_model_best_param_gsv(X_training=X_training,y_training=y_training,X_test=X_test,y_test=y_test,models=models,param=params)

            logging.info("evaluating best model name and score")
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            for k,v in model_report.items():
                v = round(v,2)
                logging.info(f"Model: {k}, R2 Score: {v}")
            logging.info(f"best model selected is {best_model_name} with accuracy of {best_model_score}")

            if best_model_score<0.5:
                logging.error("unable to find any model with 0.5 and above accuracy,exiting")
                raise CustomException("unable to find any model with 0.5 and above accuracy, exiting")

            logging.info("best model evaluated")
            logging.info("saving trained model objects")
            save_object(
                file_path=self.training_model_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info("completed training model")
            return r2_square, best_model_name
        
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)