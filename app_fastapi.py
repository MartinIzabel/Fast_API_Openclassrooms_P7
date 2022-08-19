# Library imports
import pandas as pd
import numpy as np
from fastapi import FastAPI
import joblib
from sklearn import preprocessing 
import uvicorn
from pydantic import BaseModel

#Create the app object
app = FastAPI()

# Load the model
model = joblib.load("model_vf.pkl")

@app.get("/")
def hello_world():
    return {"API" : "Prêt à Dépenser "} 

# define model for post request.
class ModelParams(BaseModel):
    data: dict

# @app.post("/predict")
# def predict(params: ModelParams):
#     return params

#Load the data
app_test = pd.read_csv('app_test.csv', sep = ",",  index_col='SK_ID_CURR')
X_test = preprocessing.StandardScaler().fit_transform(app_test)
app_test['prediction'] = (model.predict_proba(X_test)[:,1])

@app.get("/score")
def score():
    return {"prediction" : app_test['prediction']} 


#################################################################################
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.01', port=8000)
