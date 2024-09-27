#file used to create code for common functions
import os
import sys
import pickle
import dill
import pandas as pd
import numpy as np

from src.tools.custom_exception import CustomException
from src.tools.custom_logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

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
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]
            gs = GridSearchCV(model,param)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            #y_train_pred = model.predict(X_train)
            #training_model_score = r2_score(y_train, y_train_pred)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)
    

def evaluate_model_best_param_rsv(X_train, y_train,X_test,y_test,models,params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]
            gs = RandomizedSearchCV(model,param)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            #y_train_pred = model.predict(X_train)
            #training_model_score = r2_score(y_train, y_train_pred)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)
