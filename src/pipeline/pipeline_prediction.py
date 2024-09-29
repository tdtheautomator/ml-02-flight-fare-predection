#file used to create code for predection
import os
import sys
import pandas as pd
from src.tools.custom_exception import CustomException
from src.tools.custom_logger import logging
from src.tools.common import load_object



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("outputs","trained_model.pkl")
            preprocessor_path=os.path.join('outputs','encoded_data.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
