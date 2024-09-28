#file used to create code for common functions
import os
import sys
import pickle
import dill
import pandas as pd
import numpy as np

from src.tools.custom_exception import CustomException
from src.tools.custom_logger import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

#function for saving pickle file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)

 #function for loading pickle file   
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)

 #function for evaluating models for best parameters
def evaluate_model_best_param_gsv(X_train, y_train,X_test,y_test,models,params):
    logging.info("using grid search")
    performance_metrics = {}
    try:
        report = {}
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]
            gs = GridSearchCV(model,param)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            #y_train_pred = model.predict(X_train)
            #training_model_score = r2_score(y_train, y_train_pred)
            y_test_pred = model.predict(X_test)
            performance_metrics = get_model_performance_metrics(y_test, y_test_pred)
            test_model_score = performance_metrics[3]
            report[list(models.keys())[i]] = test_model_score
            logging.info(f'{model_name} | MAE : {performance_metrics[0]}, MSE : {performance_metrics[1]}, RMSE : {performance_metrics[2]}, R2 Score : {performance_metrics[3]}')
        return report
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)
    
def evaluate_model_best_param_rsv(X_train, y_train,X_test,y_test,models,params):
    logging.info("using randomized search")
    performance_metrics = {}
    try:
        report = {}
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]
            gs = RandomizedSearchCV(model,param)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            #y_train_pred = model.predict(X_train)
            #training_model_score = r2_score(y_train, y_train_pred)
            y_test_pred = model.predict(X_test)
            performance_metrics = get_model_performance_metrics(y_test, y_test_pred)
            test_model_score = performance_metrics[3]
            report[list(models.keys())[i]] = test_model_score
            logging.info(f'{model_name} | MAE : {performance_metrics[0]}, MSE : {performance_metrics[1]}, RMSE : {performance_metrics[2]}, R2 Score : {performance_metrics[3]}')
        return report
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)

def get_model_performance_metrics(true, predicted):
    logging.info("getting performance metrics")
    mae = round(mean_absolute_error(true, predicted),2)
    mse = round(mean_squared_error(true, predicted),2)
    rmse = round(root_mean_squared_error(true, predicted),2)
    r2_sc = round(r2_score(true, predicted),2)
    return mae, mse, rmse, r2_sc