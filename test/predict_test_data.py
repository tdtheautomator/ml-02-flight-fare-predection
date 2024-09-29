import pandas as pd
from src.pipeline.pipeline_prediction import PredictPipeline
import time


test_df=pd.read_csv('./outputs/test_data.csv') #update with dataset file
predict_df = test_df.copy()
predict_df.rename(columns={"price": "Actual Price"}, errors="raise", inplace=True)

call_pipeline = PredictPipeline()
predection_result = call_pipeline.predict(predict_df)
df_output=pd.DataFrame({'Predicted Price':predection_result})
df_final=predict_df.merge(df_output,left_index=True,right_index=True)

df_final["Prediction Variance"] = round(df_final['Predicted Price'] - df_final['Actual Price'],0)
filepath = f'./outputs/{time.strftime("%Y%m%d_%H%M%S")}_PredectionOutput_TestData.csv'
df_final.to_csv(filepath)