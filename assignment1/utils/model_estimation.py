import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from utils.load import load_feature_target_set

import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import math
import random
from pathlib import Path
from tqdm import tqdm
plt.style.use('seaborn')

def evaluation_mse(mod, X, y_true, plot=False, path="figures/", name="test_figure"):
    y_pred = mod.predict(X)
    if plot:
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        ax.scatter(y_pred, y_true)
        ax.legend()
        plt.savefig(path + name + ".png")
    return mean_squared_error(y_true, y_pred)

def evaluation_mae(mod, X, y_true, plot=False, path="figures/", name="test_figure"):
    y_pred = mod.predict(X)
    if plot:
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        ax.scatter(y_pred, y_true)
        ax.legend()
        plt.savefig(path + name + ".png")
    return mean_absolute_error(y_true, y_pred)

def evaluation_model_torch(mod: nn.Module, data, criterion):
    pred = mod(data.x_t)
    loss = criterion(pred, data.y_t)
    return loss.item()

def stratified_split_by_id(X_recurrent, X_simple, X_baseline, y):
    id_columns = [c for c in X_recurrent[0].columns if "id_" in c]
    df_ids = pd.DataFrame(columns = id_columns, data = [list(x.iloc[0][id_columns].values) for x in X_recurrent])
    id_series = df_ids.apply(lambda x: x.idxmax(), axis = 1)
    
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
    train_idx, test_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=.8, random_state=42).split(id_series, id_series))

    X_simple, X_recurrent, X_baseline, y = np.asarray(X_simple), np.asarray(X_recurrent), np.asarray(X_baseline), np.array(y)
    X_simple_train, X_simple_test = X_simple[train_idx], X_simple[test_idx]
    X_recurrent_train, X_recurrent_test = X_recurrent[train_idx], X_recurrent[test_idx]
    X_baseline_train, X_baseline_test = X_baseline[train_idx], X_baseline[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_simple_train, X_recurrent_train, X_baseline_train, y_train, X_simple_test, X_recurrent_test, X_baseline_test, y_test

######################
###### MODEL X #######
######################

def fit_nn():
    X_recurrent, X_simple, X_baseline, y = load_feature_target_set()
    X_train, _, _, y_train, X_test, _, _, y_test = stratified_split_by_id(X_recurrent, X_simple, X_baseline, y)
    
######################
###### MODEL RNN #######
######################
from utils.torch_utils import train_torch_model, cross_validate_torch, LSTM, evaluation_model_torch_numpy_mse, DataSet
    
def fit_rnn():
    X_recurrent, X_simple, X_baseline, y = load_feature_target_set()
    _, X_train, _, y_train, _, X_test, _, y_test = stratified_split_by_id(X_recurrent, X_simple, X_baseline, y)
    n_features = X_train[0].shape[1]
    
    
    # gridsearch_results = pd.DataFrame(columns = ["lr", "batch_size", "hid_dim", "epochs", "avg_score"])
    # k = 0
    # for lr in [0.001]:
    #     for batch_size in [8, 10, 12]:
    #         for hid_dim in [50]:
    #             for epochs in [10, 15, 25]:
    #                 params = {
    #                     "lr": lr,
    #                     "batch_size": batch_size,
    #                     "hid_dim": hid_dim,
    #                     "epochs": epochs
    #                 }
    #                 fold_scores_train, fold_scores_test = cross_validate_torch(X_train, y_train, verbose = 0, **params)
    #                 print(params, fold_scores_test)
    #                 gridsearch_results.loc[k] = [lr, batch_size, hid_dim, epochs, fold_scores_test]
    #                 k+=1
    
    # if cross_val:
    #     pd.argam
    # else:
        
    
    data_train = DataSet(X_train, y_train)
    data_test = DataSet(X_test, y_test)
    
    model = LSTM(in_dim=n_features, hid_dim=50, out_dim=1)
    model, train_losses, test_losses = train_torch_model(
        model = model,
        optimizer = torch.optim.Adam(model.parameters(), lr=.001),
        criterion = nn.MSELoss(),
        data_train = data_train,
        data_validate = data_test,
        epochs = 50
    )
    best_loss_test = test_losses[-1]
    print(f"RNN - Test - MSE loss = {best_loss_test}")
    # mse = evaluation_model_torch_numpy_mse(model, data_test)
    # print(f"mse testset: {mse}")

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

def fit_lgb():
    X_recurrent, X_simple, X_baseline, y = load_feature_target_set()
    columns = X_simple.columns
    X_train, _, _, y_train, X_test, _, _, y_test = stratified_split_by_id(X_recurrent, np.array(X_simple), X_baseline, y)
    print("Shape of training features: ", X_train.shape)
    print("Shape of testing features: ", X_test.shape)
    
    print("Performing Kfolds for LGMBoost ...")
    param_grid = {
        # "num_leaves": [],
        "learning_rate": [0.1, 0.05]
    }
    mod = lgb.LGBMModel(objective="regression")
    gs = GridSearchCV(
        estimator = mod,
        param_grid = param_grid,
        cv = KFold(10, shuffle=True),
        scoring = evaluation_mse,
        verbose=0
    )
    gs.fit(pd.DataFrame(columns = columns, data = X_train), y_train)
    y_pred = gs.predict(X_test)
    # lgb.plot_importance(gs.best_estimator_, )
    # plt.show()
    mse = mean_squared_error(y_test, y_pred)
    print(f"LGBM - GridSearch - Best params = {gs.best_params_} - Best avg score = {gs.best_score_}")
    print(f"LGBM - Test - MSE loss = {mse}")

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
    y_pred = X_test.copy()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # import matplotlib.pyplot as plt
    # plt.hist(np.array(y_test).reshape(-1) - np.array(y_pred).reshape(-1))
    # plt.show()
    # plt.scatter(np.array(y_pred).reshape(-1), np.array(y_test).reshape(-1))
    # plt.show()
    print(f"Baseline - Test - MSE loss = {mse} | mae: {mae}")
    
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
    "rnn": fit_rnn,
}