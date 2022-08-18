# Library imports
from fastapi import FastAPI
import joblib
import pandas as pd
from sklearn import preprocessing 
import uvicorn

#Create the app object
app = FastAPI()

#Load the data
app_test = pd.read_csv('app_test.csv', sep = ",",  index_col='SK_ID_CURR')
X_test = preprocessing.StandardScaler().fit_transform(app_test)

# # Load the model
model = joblib.load("model_vf.pkl")

# #Preparation des predictions
# seuil = 0.7539816036060938
# app_test['prediction'] = (model.predict_proba(X_test)[:,1])
# app_test_bool = app_test
# app_test_bool['prediction_label'] =  (model.predict_proba(X_test)[:,1] > seuil).astype(bool)

@app.get("/")
def hello_world():
    return {"hello" : "world"} 

@app.get('/prediction/{id}')
def predict_model(id):
    print(app_test[0])
    return {"id": id}

@app.get('/client_data/{id}')
def get_data(id):
    return {app_test.loc[id]}

@app.get('/lists_feat')
def df_print():
    return {app_test[0]}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.01', port=8000)
