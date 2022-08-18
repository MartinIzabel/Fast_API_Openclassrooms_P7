# Library imports
from fastapi import FastAPI
import joblib
import pandas as pd
from sklearn import preprocessing 
import uvicorn

#Create the app object
app = FastAPI()

#Load the data
# app_test = pd.read_csv('app_test.csv', sep = ",",  index_col='SK_ID_CURR')

# # Load the model
model = joblib.load("model_vf.pkl")

# #listes des features
# features = list(app_test.columns)

# #Preparation des predictions
# seuil = 0.7539816036060938
# X_test = preprocessing.StandardScaler().fit_transform(app_test)
# app_test['prediction'] = (model.predict_proba(X_test)[:,1])
# app_test_bool = app_test
# app_test_bool['prediction_label'] =  (model.predict_proba(X_test)[:,1] > seuil).astype(bool)

@app.get("/")
def hello_world():
    return {"hello" : "world"} 

@app.get('/prediction/{id}')
def predict_model(id):
    return {"id": id}

@app.get('/prediction_bool')
def predict_bool_model():
    return {"10001":"1", "100002":"2"}

@app.get('/lists_feat')
def predict_bool_model():
    return model

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.01', port=8000)
