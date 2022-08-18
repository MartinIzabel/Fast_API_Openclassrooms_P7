# Library imports
from array import array
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
seuil = 0.7539816036060938

def get_prediction(param):
    
    x = param

    y = model.predict(x)[0]  # just get single value
    prob = model.predict_proba(x)[0].tolist()  # send to list for return

    return {'prediction': int(y), 'probability': prob}


# app_test['prediction'] = (model.predict_proba(X_test)[:,1])
# app_test_bool = app_test
# app_test_bool['prediction_label'] =  (model.predict_proba(X_test)[:,1] > seuil).astype(bool)

@app.get("/")
def hello_world():
    return {"hello" : "world"} 

@app.get('/prediction/{id}')
def predict_model(id):
    return {seuil: id}

# define model for post request.
# class ModelParams(BaseModel):
#     params: float

@app.post("/predict")
def predict(params):

    pred = get_prediction(params)

    return pred




#################################################################################
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.01', port=8000)
