import time,os,re,csv,sys,uuid,joblib
from datetime import date
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from logger2 import update_predict_log, update_train_log
from dataingestion2 import fetch_ts, engineer_features

## model specific variables (iterate the version and note with each change)
localpath = os.path.abspath('')
data_dir = os.path.join(localpath,"data_dir")
from sklearn import linear_model

def model1_RandomForest(data_dir):
    ts_data = fetch_ts(data_dir)
    df = ts_data['all']
    X,y,dates = engineer_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        shuffle=True, random_state=42)
    param_grid_rf = {
    'rf__criterion': ['mse','mae'],
    'rf__n_estimators': [10,15,20,25]
    }

    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                              ('rf', RandomForestRegressor())])
    
    grid = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=5, iid=False, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    eval_rmse =  round(np.sqrt(mean_squared_error(y_test,y_pred)))
    return(eval_rmse)

def model2_Linearregression(data_dir):
    ts_data = fetch_ts(data_dir)
    df = ts_data['all']
    X,y,dates = engineer_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        shuffle=True, random_state=42)
    
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    eval_rmse =  round(np.sqrt(mean_squared_error(y_test,y_pred)))
    return(eval_rmse)

##Comparision between Linear Regression & Random Forest for least RMSE error.
model1_rmse = model1_RandomForest(data_dir)
model2_rmse = model2_Linearregression(data_dir)
if model1_rmse < model2_rmse:
    print("Random Forest is the best model for given data")
else:
    print("Linear Regression is the best model for given data")
