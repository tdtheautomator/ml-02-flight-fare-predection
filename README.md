# Machine Learning Flight Fare Predection

Sample dataset from [Kaggle](https://www.kaggle.com/)<br />
Dataset Shape : Rows : 300153 , Columns : 10<br />
- numerical features : 3 : ['duration', 'days_left', 'price']
- categorical features : 7 : ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']
- training dataset is 80%

## Models Used

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