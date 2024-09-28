#file used to create code for transforming data
import os
import sys
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.tools.custom_exception import CustomException
from src.tools.custom_logger import logging
from src.tools.common import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('outputs',"encoded_data.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        logging.info("initiated data transformation")
        try:
            logging.info("defining numeric columns")
            numeric_columns = [
                'duration', 
                'days_left'
                ]

            logging.info("defining category columns")
            category_columns = [
                'airline', 
                'flight', 
                'source_city', 
                'departure_time', 
                'stops', 
                'arrival_time', 
                'destination_city', 
                'class'
                ]
            
            logging.info("defining numeric pipeline")
            numeric_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")), #handle missing values
                ("scaler",StandardScaler())
                ]
            )
            logging.info("defining category pipeline")
            category_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")), #handle missing values
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("initiating column transformer")
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",numeric_pipeline,numeric_columns),
                ("cat_pipelines",category_pipeline,category_columns)
                ]
            )
            
            return preprocessor

        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,training_data_path,test_data_path):
        try:
            logging.info("loading training data into dataframe")
            training_df=pd.read_csv(training_data_path)
            logging.info('loading test data into dataframe')
            test_df=pd.read_csv(test_data_path)

            logging.info("obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name = "price" #update with column to be predicted
            
            logging.info(f"dropping target column {target_column_name} from training dataframe" )
            input_feature_training_df=training_df.drop(columns=[target_column_name],axis=1)
            target_feature_training_df=training_df[target_column_name]

            logging.info(f"dropping target column {target_column_name} from test dataframe" )
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("performing preprocessing")
            '''
            fit() method helps in fitting the data into a model.
            transform() method helps in transforming the data into a form that is more suitable for the model.
            fit_transform() method combines the functionalities of both fit() and transform() methods in one step.
            '''
            input_feature_training_arr=preprocessing_obj.fit_transform(input_feature_training_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("slicing objects to concatenation along the second axis")
            training_arr = np.c_[input_feature_training_arr, np.array(target_feature_training_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("saving preprocessing objects")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("data transformation completed")
            return (
                training_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)