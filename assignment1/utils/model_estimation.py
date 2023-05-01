import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from utils.load import load_feature_target_set
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.style.use('seaborn')

###############################
###### HELPER FUNCTIONS #######
###############################

def evaluation_mse_score(mod, X, y_true, plot=False, path="figures/", name="test_figure"):
    y_pred = mod.predict(X)
    if plot:
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        ax.scatter(y_pred, y_true)
        ax.legend()
        plt.savefig(path + name + ".png")
    return -mean_squared_error(y_true, y_pred)

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
    """ the ids list containes the id of each user for each feature/target pair. Must be same length as features/targets """
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
    train_idx, test_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=.8, random_state=6).split(ids, ids))

    X_simple, X_recurrent, X_baseline, y = np.asarray(X_simple), np.asarray(X_recurrent), np.asarray(X_baseline), np.array(y)
    X_simple_train, X_simple_test = X_simple[train_idx], X_simple[test_idx]
    X_recurrent_train, X_recurrent_test = X_recurrent[train_idx], X_recurrent[test_idx]
    X_baseline_train, X_baseline_test = X_baseline[train_idx], X_baseline[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_simple_train, X_recurrent_train, X_baseline_train, y_train, X_simple_test, X_recurrent_test, X_baseline_test, y_test

########################
###### MODEL RNN #######
########################
from utils.torch_utils import train_torch_model, cross_validate_torch, LSTM, evaluation_model_torch_numpy_mse, evaluation_model_torch_numpy_mae, DataSet
    
def fit_rnn(args):
    X_recurrent, X_simple, X_baseline, y, ids = load_feature_target_set()
    _, X_train, _, y_train, _, X_test, _, y_test = stratified_split_by_id(X_recurrent, X_simple, X_baseline, y, ids)
    n_features = X_train[0].shape[1]
    
    # turn off once correct specification determined
    gridsearch = True
    
    if gridsearch == True:
        param_names = ["lr", "batch_size", "hid_dim", "epochs"]
        columns = param_names + ["avg_score"]
        gridsearch_results = pd.DataFrame(columns = columns)
        k = 0
        for lr in [0.01]:
            for batch_size in [10]:
                for hid_dim in [5, 15, 25, 50]:
                    for epochs in [5,10,20,40]:
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
                        print(params, fold_scores_test)
                        gridsearch_results.loc[k] = [lr, batch_size, hid_dim, epochs, fold_scores_test]
                        k+=1 
        lr, batch_size, hid_dim, epochs, _ = tuple(gridsearch_results.sort_values("avg_score").reset_index(drop=True).iloc[0].values)
        batch_size, epochs, hid_dim = int(batch_size), int(epochs), int(hid_dim)
        
        print(f"RNN - GridSearch - Best params = {dict(zip(columns, (lr, batch_size, hid_dim, epochs)))} - Best avg sore = {_}")
    else:
        # default parameters
        lr = .01
        batch_size = 10
        hid_dim = 5
        epochs = 40
    
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
    
    y_pred = model(data_test.x_t).detach().numpy().reshape(-1,)
    y_true = data_test.y_t.detach().numpy().reshape(-1,)
    
    print(f"RNN - Test - MSE loss = {best_loss_test} - MAE loss = {mae}")
    
    return y_pred, y_true
    
########################
###### MODEL LGB #######
########################

def fit_lgb(args):
    """ Determine the model specification through a cross validated grid search"""
    X_recurrent, X_simple, X_baseline, y, ids = load_feature_target_set()
    columns = X_simple.columns
    X_train, _, _, y_train, X_test, _, _, y_test = stratified_split_by_id(X_recurrent, np.array(X_simple), X_baseline, y, ids)
    
    gridsearch = False
    if gridsearch:
        param_grid = {
            # "num_leaves": [],
            "learning_rate": [0.16, 0.08, 0.04, 0.02, 0.01],
            "max_depth": [-1, 1, 2, 4],
            "num_leaves": [5, 10, 20, 40, 80],
            "n_estimators": [50, 100],
            "min_split_gain": [0, 1]
        }
    else:
        param_grid = {
            # "num_leaves": [],
            "learning_rate": [0.02,],
            "max_depth": [-1,],
            "num_leaves": [20,],
            "n_estimators": [100]
        }
    mod = lgb.LGBMModel(objective="regression")
    gs = GridSearchCV(
        estimator = mod,
        param_grid = param_grid,
        cv = KFold(5, shuffle=True, random_state=52),
        scoring = evaluation_mse_score,
        verbose=0
    )
    gs.fit(pd.DataFrame(columns = columns, data = X_train), y_train)
    y_pred = gs.predict(X_test)
    
    # REMOVE THIS
    lgb.plot_importance(gs.best_estimator_, importance_type="gain")
    plt.savefig("figures/lgb_importance.png", dpi = 250, bbox_inches = "tight")
    plt.clf()
    # REMOVE THIS
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"LGBM - GridSearch - Best params = {gs.best_params_} - Best avg score = {gs.best_score_}")
    print(f"LGBM - Test - MSE loss = {mse} - MAE loss = {mae}")
    
    return y_pred, y_test
    
#############################
###### MODEL BASELINE #######
#############################
def fit_baseline(args):
    """ the baseline model is an AR(1) with lagged coefficient = 1 """
    X_recurrent, X_simple, X_baseline, y, ids = load_feature_target_set()
    _, _, X_train, y_train, _, _, X_test, y_test = stratified_split_by_id(X_recurrent, X_simple, X_baseline, y, ids)
    y_pred = X_test.copy()
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Baseline - Test - MSE loss = {mse} - MAE loss = {mae}")
    
    return y_pred, y_test
    
##########################
###### MODEL ARIMA #######
##########################
def fit_arima(args):
    """ the baseline model is an AR(1) with coefficient = 1 """
    X_recurrent, X_simple, X_baseline, y, ids = load_feature_target_set()
    _, X_train, _, y_train, _, X_test, _, y_test = stratified_split_by_id(X_recurrent, X_simple, X_baseline, y, ids)
    
    print(X_train.shape)
    X_recurrent = np.array([x["mean_mood"] for x in X_train]).reshape(len(X_train), len(X_train[0]))
    # TODO add ols
    import statsmodels.api as sm
    mod = sm.OLS(y_train, X_train)
    y_pred = mod.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Baseline - Test - MSE loss = {mse} - MAE loss = {mae}")
    
#######################
## GENERAL FUNCTIONS ##
#######################
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, mae

model_functions = {
    "baseline": fit_baseline,
    "lgb": fit_lgb,
    "rnn": fit_rnn,
    "arima": fit_arima
}