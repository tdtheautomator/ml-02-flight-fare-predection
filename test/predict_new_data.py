import pandas as pd
from src.pipeline.pipeline_prediction import PredictPipeline
import time


new_df=pd.read_csv('./test/new_data.csv') #update with dataset file
call_pipeline = PredictPipeline()
predection_result = call_pipeline.predict(new_df)
df_output=pd.DataFrame({'Predicted Price':predection_result})
df_final=new_df.merge(df_output,left_index=True,right_index=True)

filepath = f'./outputs/{time.strftime("%Y%m%d_%H%M%S")}_PredectionOutput_NewData.csv'
df_final.to_csv(filepath)
print("-*40")
print("Predection Results\n")
print(df_final)
print("-*40")