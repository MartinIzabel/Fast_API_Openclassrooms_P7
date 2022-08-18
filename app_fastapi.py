# Library imports
from fastapi import FastAPI
import joblib
import pandas as pd
from sklearn import preprocessing 

#Create the app object
app = FastAPI()

#Load the data
app_test = pd.read_csv('app_test.csv', sep = ",",  index_col='SK_ID_CURR')

# # Load the model
model = joblib.load("model_vf.pkl")

# #listes des features
features = list(app_test.columns)

# #Preparation des predictions
seuil = 0.7539816036060938
X_test = preprocessing.StandardScaler().fit_transform(app_test)
prediction = (model.predict_proba(X_test)[:,1])
app_test_bool = app_test
prediction_bool =  (model.predict_proba(X_test)[:,1] > seuil).astype(bool)

@app.get("/")
def hello_world():
    return {"hello" : "world"} 

@app.get('/prediction/')
def predict_model():
    return {prediction}

@app.get('/prediction_bool/')
def predict_bool_model():
    return {prediction_bool}
