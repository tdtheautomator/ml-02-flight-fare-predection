# Machine Learning Flight Fare Predection

Sample dataset from [Kaggle](https://www.kaggle.com/)<br />
Dataset Shape : Rows : 300153 , Columns : 10<br />
- numerical features : 3 : ['duration', 'days_left', 'price']
- categorical features : 7 : ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']
- training dataset is 80%

## Models Used
Model training is performed using multiple regression, ensemble, tree techniques. Models are configured with hyper tuning parameters and used GridSearch to identify best paramaeters<br />
Model with lowest mean square error and highest R2 Score is selected for final training<br />
Outputs for ecnoding and model training is saved in a pickle file<br />
Once the model is trained only trained_model.pkl file is needed to perform predictions<br />

- Linear Regression
- Catagory Boost Regressor
- Bagging Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor

## Usage (CLI)

- Ensure anaconda is installed [Anaconda Download](https://www.anaconda.com/download)
- Clone git repo
- Create new virtual environment
```
conda create -p venv python==3.12
```
- Activate new virtual environment
```
conda activate ./venv
```
- Deploy requirements
```
pip install -r requirements.txt
```
- Start data ingestion, transformation and training
```
python .\src\components\data_ingestion.py
```
- Predict with test data
```
python ./test/predict_test_data.py
```
- Predict with new data (flight from mumbai to delhi booked 7 days and 14 days in advance)
```
python ./test/predict_new_data.py
```

## Notes
- Detailed logging is available under ./logs
- Output files are saved in ./outputs

## Known Issues
 - Model performnace can be further  fine tuned by converting some columns from categorical to numerica. example : stops column can be converted to numerical.
 - Missing Handler for unwanted columns. example duration is not needed as an input and can use stops<br />
 - Data for certain predections is highly skewed, further tuning is needed<br />
