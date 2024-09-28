#file used to create code for ingesting data
import os
import sys
from src.tools.custom_exception import CustomException
from src.tools.custom_logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.training_model import TrainingModel, TrainingModelConfig
import time


@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('outputs',"raw_data.csv")
    training_data_path: str=os.path.join('outputs',"training_data.csv")
    test_data_path: str=os.path.join('outputs',"test_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("initiated data ingestion")
        try:
            #curr_dir=os.getcwd()
            #logging.info(f"current directory is {curr_dir}")
            logging.info("reading datset as dataframe")
            df=pd.read_csv('./data/clean_dataset.csv') #update with dataset file
            df.drop(['Unnamed: 0'],axis=1, inplace = True) #drop unwanted column permanently
            numerical_features = [feature for feature in df.columns if df[feature].dtype != 'O']
            categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
            
            logging.info(f'Number of Rows: {df.shape[0]}')
            logging.info(f'Number of Columns: {df.shape[1]}')
            logging.info(f'Numerical Features : {format(len(numerical_features))} : {numerical_features}')
            logging.info(f'Numerical Features : {format(len(categorical_features))} : {categorical_features}')
            
            logging.info("creating outputs folders if doesn't exist")
            os.makedirs(os.path.dirname(self.ingestion_config.training_data_path),exist_ok=True)
            
            logging.info("exporting dataframe to csv to outputs")
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("spliting training and test data")
            training_set,test_set=train_test_split(df,test_size=0.20,random_state=42)
            
            logging.info("exporting training dataset to outputs")
            training_set.to_csv(self.ingestion_config.training_data_path,index=False,header=True)
            
            logging.info("exporting test dataset to outputs")
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("data ingestion completed")
            #print(type(training_set))
            #print(type(test_set))

            return(
                self.ingestion_config.training_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)

if __name__=="__main__":
    print('-'*80)
    print("execution started")

    data_ingestion_start_time = time.time()
    obj=DataIngestion()
    training_data,test_data=obj.initiate_data_ingestion()
    data_ingestion_end_time = time.time()
    data_ingestion_time = round((data_ingestion_end_time - data_ingestion_start_time),2)
    logging.info(f"data ingestion completed in {data_ingestion_time} secs")
    print(f" - data ingestion completed in {data_ingestion_time} secs")

    data_transformation=DataTransformation()
    training_arr,test_arr,_=data_transformation.initiate_data_transformation(training_data,test_data)
    data_transformation_end_time = time.time()
    data_transformation_time = round((data_transformation_end_time - data_ingestion_end_time),2)
    logging.info(f"data transformation completed in {data_transformation_time} secs")
    print(f" - data transformation completed in {data_transformation_time} secs")
#
#    modeltrainer=TrainingModel()
#    results = modeltrainer.initiate_training_model(training_arr,test_arr)
#    accuracy = round(results[0]*100,2)
#    training_model_end_time = time.time()
#    training_model_time = round((training_model_end_time - data_transformation_end_time),2)
#    total_time = round((training_model_end_time - data_ingestion_start_time),2)
#    logging.info(f"training model completed in {training_model_time} secs")
#    logging.info(f"execution completed in {training_model_time} secs with best accuracy of {accuracy}% using {results[1]} model")
#    print(f" - training model completed in {training_model_time} secs")
#    print('-'*80)
#    print(f"execution completed in {training_model_time} secs with best accuracy of {accuracy}% using {results[1]} model")
#    print('-'*80)
