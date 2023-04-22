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
import matplotlib.pyplot as plt
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

def stratified_split_by_id(X_recurrent, X_simple, X_baseline, y, ids: list):
    """ ids containes the id of each user for each feature/target pair. Must be same length as features/targets """
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
    train_idx, test_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=.8, random_state=42).split(ids, ids))

    X_simple, X_recurrent, X_baseline, y = np.asarray(X_simple), np.asarray(X_recurrent), np.asarray(X_baseline), np.array(y)
    X_simple_train, X_simple_test = X_simple[train_idx], X_simple[test_idx]
    X_recurrent_train, X_recurrent_test = X_recurrent[train_idx], X_recurrent[test_idx]
    X_baseline_train, X_baseline_test = X_baseline[train_idx], X_baseline[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_simple_train, X_recurrent_train, X_baseline_train, y_train, X_simple_test, X_recurrent_test, X_baseline_test, y_test

######################
###### MODEL RNN #######
######################
from utils.torch_utils import train_torch_model, cross_validate_torch, LSTM, evaluation_model_torch_numpy_mse, evaluation_model_torch_numpy_mae, DataSet
    
def fit_rnn(args):
    X_recurrent, X_simple, X_baseline, y, ids = load_feature_target_set()
    _, X_train, _, y_train, _, X_test, _, y_test = stratified_split_by_id(X_recurrent, X_simple, X_baseline, y, ids)
    n_features = X_train[0].shape[1]
    
    gridsearch = True
    
    if gridsearch == True:
        param_names = ["lr", "batch_size", "hid_dim", "epochs"]
        columns = param_names + ["avg_score"]
        gridsearch_results = pd.DataFrame(columns = columns)
        k = 0
        for lr in [0.001]:
            for batch_size in [10]:
                for hid_dim in [50]:
                    for epochs in [30]:
                        params = {
                            "lr": lr,
                            "batch_size": batch_size,
                            "hid_dim": hid_dim,
                            "epochs": epochs
                        }
                        fold_scores_train, fold_scores_test = cross_validate_torch(
                            X_train, y_train, 
                            verbose = 0, 
                            cv = KFold(n_splits=5, shuffle=True, random_state=52),
                            **params
                        )
                        gridsearch_results.loc[k] = [lr, batch_size, hid_dim, epochs, fold_scores_test]
                        k+=1   
        lr, batch_size, hid_dim, epochs, _ = tuple(gridsearch_results.sort_values("avg_score").reset_index(drop=True).iloc[0].values)
        batch_size, epochs, hid_dim = int(batch_size), int(epochs), int(hid_dim)
        
        print(f"RNN - GridSearch - Best params = {dict(zip(columns, (lr, batch_size, hid_dim, epochs)))} - Best avg sore = {_}")
    else:
        # default parameters
        lr = .001
        batch_size = 1
        hid_dim = 50
        epochs = 15
    
    data_train = DataSet(X_train, y_train)
    data_test = DataSet(X_test, y_test)
    
    # after cross validation or using default params, train model on full training set and evaluate on the unseen test set
    model = LSTM(in_dim=n_features, hid_dim=hid_dim, out_dim=1)
    model, train_losses, test_losses = train_torch_model(
        model = model,
        optimizer = torch.optim.Adam(model.parameters(), lr=lr),
        criterion = nn.MSELoss(),
        data_train = data_train,
        data_validate = data_test,
        epochs = epochs
    )
    best_loss_test = test_losses[-1]
    
    mae = evaluation_model_torch_numpy_mae(model, data_test)
    print(f"RNN - Test - MSE loss = {best_loss_test} - MAE loss = {mae}")

######################
###### MODEL XGB #######
######################
def fit_xgb(args):
    _, X, _, y, ids = load_feature_target_set()
    print("Loaded data for XGB ... ")
    
    print("Performing Kfolds for XGBoost")
    
######################
###### MODEL LGB #######
######################

def fit_lgb(args):
    X_recurrent, X_simple, X_baseline, y, ids = load_feature_target_set()
    columns = X_simple.columns
    X_train, _, _, y_train, X_test, _, _, y_test = stratified_split_by_id(X_recurrent, np.array(X_simple), X_baseline, y, ids)
    param_grid = {
        # "num_leaves": [],
        "learning_rate": [0.1],
        # "max_depth": [None, 5, 10],
        # "num_leaves": [100, 40, 65],
        # "n_estimators": [20, 50, 100, 200]
        
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
    # lgb.plot_importance(gs.best_estimator_, importance_type="gain")
    # plt.show()
    mse = mean_squared_error(y_test, y_pred)
    print(f"LGBM - GridSearch - Best params = {gs.best_params_} - Best avg score = {gs.best_score_}")
    print(f"LGBM - Test - MSE loss = {mse}")

######################
###### MODEL z #######
######################
def fit_svr(args, random_state=42):
    
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
def fit_baseline(args):
    """ the baseline model is an AR(1) with coefficient = 1 """
    X_recurrent, X_simple, X_baseline, y, ids = load_feature_target_set()
    _, _, X_train, y_train, _, _, X_test, y_test = stratified_split_by_id(X_recurrent, X_simple, X_baseline, y, ids)
    y_pred = X_test.copy()
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Baseline - Test - MSE loss = {mse} | MAE loss = {mae}")
    
#######################
## GENERAL FUNCTIONS ##
#######################
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, mae

model_functions = {
    "xgb": fit_xgb,
    "svr": fit_svr,
    "baseline": fit_baseline,
    "lgb": fit_lgb,
    "rnn": fit_rnn,
}