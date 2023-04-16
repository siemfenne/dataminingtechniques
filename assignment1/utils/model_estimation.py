import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from utils.load import load_feature_target_set

######################
###### MODEL X #######
######################
def fit_nn():
    X, y = load_feature_target_set()

######################
###### MODEL Y #######
######################
def fit_xgb():
    X, y = load_feature_target_set()

######################
###### MODEL z #######
######################
def fit_svr(random_state=42):
    
    X, y = load_feature_target_set()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, shuffle=True, random_state=random_state)
    
    param_grid = {
        "C": [1e1, 1e0, 1e-1, 1e-2],
    }
    grid_search = GridSearchCV(estimator=SVR(), cv=KFold(n_splits=5), param_grid=param_grid)
    grid_search.fit(X_train, y_train)
    best_params, scores = grid_search.best_params_
    
    mse, mae = evaluate_model(grid_search, X_test, y_test)
    
    return {
        "best_params": best_params,
        "scores": scores,
        "mse": mse,
        "mae": mae
    }
    
    
######################
###### MODEL z #######
######################
def fit_baseline():
    """ the baseline model is an AR(1) with coefficient = 1 """
    X, y = load_feature_target_set("data/")
    # X = y.shift(1)
    
#######################
## GENERAL FUNCTIONS ##
#######################
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, mae

model_functions = {
    "nn": fit_nn,
    "xgb": fit_xgb,
    "svr": fit_svr,
    "baseline": fit_baseline
}