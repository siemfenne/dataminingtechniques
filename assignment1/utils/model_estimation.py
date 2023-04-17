import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from utils.load import load_feature_target_set

def evaluation_mse(mod, X, y_true):
    y_pred = mod.predict(X)
    return mean_squared_error(y_true, y_pred)

def stratified_split_by_id(X_recurrent, X_simple, X_baseline, y):
    import random
    from random import shuffle
    random.seed(5)
    
    number_of_ids = len([c for c in X_recurrent[0].columns if "id_" in c])
    dummy_ids = np.asarray([list(x.iloc[0][[c for c in x.columns if "id_" in c]]) for x in X_recurrent]).reshape(-1,number_of_ids)
    
    blocks, _ = np.unique(dummy_ids, return_inverse=True)
    block_count = np.bincount(_)
    where = np.argsort(_)
    block_start = np.concatenate(([0], np.cumsum(block_count)[:-1]))

    x = 4/5 + 3/5/(block_count - 2)
    x = np.clip(x, 0, 1) # if n in (2, 3), the ratio is larger than 1
    threshold = np.repeat(x, block_count)
    threshold[block_start] = 1 # first item goes to A
    threshold[block_start + 1] = 0 # seconf item goes to B

    idx = threshold > np.random.rand(len(dummy_ids))
    print(idx, len(idx))
    X_simple_train, X_simple_test = X_simple[where[idx]], X_simple[~where[idx]]
    X_recurrent_train, X_recurrent_test = X_recurrent[where[idx]], X_recurrent[~where[idx]]
    X_baseline_train, X_baseline_test = X_baseline[where[idx]], X_baseline[~where[idx]]
    y_train, y_test = y[where[idx]], y[~where[idx]]
    return X_simple_train, X_recurrent_train, X_baseline_train, y_train, X_simple_test, X_recurrent_test, X_baseline_test, y_test

######################
###### MODEL X #######
######################
def fit_nn():
    X_recurrent, X_simple, X_baseline, y = load_feature_target_set()
    X_train, _, _, y_train, X_test, _, _, y_test = stratified_split_by_id(X_recurrent, X_simple, X_baseline, y)

######################
###### MODEL XGB #######
######################
def fit_xgb():
    _, X, _, y = load_feature_target_set()
    print("Loaded data for XGB ... ")
    
    print("Performing Kfolds for XGBoost")
    
######################
###### MODEL LGB #######
######################

def evaluation_mae(mod, X, y_true):
    y_pred = mod.predict(X)
    return mean_absolute_error(y_true, y_pred)

def fit_lgb():
    X_recurrent, X_simple, X_baseline, y = load_feature_target_set()
    X_train, _, _, y_train, X_test, _, _, y_test = stratified_split_by_id(X_recurrent, X_simple, X_baseline, y)
    print("Shape of training features: ", X_train.shape)
    print("Shape of testing features: ", X_test.shape)
    
    print("Performing Kfolds for LGMBoost")
    param_grid = {
        # "num_leaves": [],
        "learning_rate": [0.1, 0.05, 0.01]
    }
    mod = lgb.LGBMModel(objective="regression")
    gs = GridSearchCV(
        estimator = mod,
        param_grid = param_grid,
        cv = KFold(10, shuffle=True),
        scoring = evaluation_mse,
        verbose=10
    )
    gs.fit(X_train, y_train)
    
    import matplotlib.pyplot as plt
    y_pred = gs.predict(X_test)
    plt.hist(np.array(y_pred).reshape(-1) - np.array(y_test).reshape(-1))
    plt.show()
    print("results: ", gs.best_score_, gs.best_params_)

######################
###### MODEL z #######
######################
def fit_svr(random_state=42):
    
    _, X_train, _, y_train, _, X_test, _, y_test = stratified_split_by_id(load_feature_target_set())    
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
    X_recurrent, X_simple, X_baseline, y = load_feature_target_set()
    _, _, X_train, y_train, _, _, X_test, y_test = stratified_split_by_id(X_recurrent, X_simple, X_baseline, y)
    X: np.ndarray
    y_pred = X.copy()
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    import matplotlib.pyplot as plt
    plt.hist(np.array(y).reshape(-1) - np.array(y_pred).reshape(-1))
    plt.show()
    print(f"mse: {mse} | mae: {mae}")
    
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
    "baseline": fit_baseline,
    "lgb": fit_lgb,
}