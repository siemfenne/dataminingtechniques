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

def evaluation_mse(mod, X, y_true):
    y_pred = mod.predict(X)
    return mean_squared_error(y_true, y_pred)

def evaluation_model_torch(mod: nn.Module, data, criterion):
    pred = mod(data.x_t)
    loss = criterion(pred, data.y_t)
    return loss.item()

def stratified_split_by_id(X_recurrent, X_simple, X_baseline, y):
    id_columns = [c for c in X_recurrent[0].columns if "id_" in c]
    df_ids = pd.DataFrame(columns = id_columns, data = [list(x.iloc[0][id_columns].values) for x in X_recurrent])
    id_series = df_ids.apply(lambda x: x.idxmax(), axis = 1)
    
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
    for (train_idx, test_idx) in StratifiedShuffleSplit(n_splits=1, train_size=.8, random_state=42).split(id_series, id_series):
        pass
    # train_idx, test_idx = list(train_idx).astype(int), list(test_idx).astype(int)

    X_simple, X_recurrent, X_baseline, y = np.asarray(X_simple), np.asarray(X_recurrent), np.asarray(X_baseline), np.array(y)
    X_simple_train, X_simple_test = X_simple[train_idx], X_simple[test_idx]
    X_recurrent_train, X_recurrent_test = X_recurrent[train_idx], X_recurrent[test_idx]
    X_baseline_train, X_baseline_test = X_baseline[train_idx], X_baseline[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_simple_train, X_recurrent_train, X_baseline_train, y_train, X_simple_test, X_recurrent_test, X_baseline_test, y_test

######################
###### MODEL X #######
######################

class LSTM(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, window_size = 5):
        super(LSTM, self).__init__()
        self.hid_dim = hid_dim
        self.window_size = window_size

        self.lstm = nn.LSTM(in_dim, hid_dim)
        self.linear = nn.Linear(hid_dim, out_dim)
        self.hidden_cell = (torch.zeros(1,self.window_size,self.hid_dim),
                            torch.zeros(1,self.window_size,self.hid_dim))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out)
        self.init_state()
        return predictions[-1]
    
    def init_state(self):
        self.hidden_cell = (torch.zeros(1,self.window_size,self.hid_dim),
                            torch.zeros(1,self.window_size,self.hid_dim))

def fit_nn():
    X_recurrent, X_simple, X_baseline, y = load_feature_target_set()
    X_train, _, _, y_train, X_test, _, _, y_test = stratified_split_by_id(X_recurrent, X_simple, X_baseline, y)
    
######################
###### MODEL RNN #######
######################
from torch.utils.data import Dataset, DataLoader

class DataSet(Dataset):
    """ Loads the x,y data into a Dataset instance as torch tensors """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x_t = torch.tensor(x, dtype=torch.float32)
        self.y_t = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self,):
        return len(self.y_t)
    
    def __getitem__(self, idx: int):
        return self.x_t[idx], self.y_t[idx]
    
    def split(self, train_index: list, test_index: list):
        """ Build-in function to split data on given training and testing indices """
        data_train = DataSet(self.x_t[train_index], self.y_t[train_index])
        data_test = DataSet(self.x_t[test_index], self.y_t[test_index])
        return data_train, data_test
    
def fit_rnn():
    X_recurrent, X_simple, X_baseline, y = load_feature_target_set()
    _, X_train, _, y_train, _, X_test, _, y_test = stratified_split_by_id(X_recurrent, X_simple, X_baseline, y)
        
    n_features = X_train[0].shape[1]
    kf = KFold(10, shuffle=True, random_state=52)
    epochs = 10
    hid_dim = 50
    learning_rate = .001
    batch_size = 10

    train_data = DataSet(X_train, y_train)
    res_fold_train = {}
    res_fold_test = {}
    
    for i, (train_idx, validate_idx) in enumerate(kf.split(train_data.x_t), 1):

        model = LSTM(in_dim=n_features, hid_dim=hid_dim, out_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        data_train, data_validate = train_data.split(train_idx, validate_idx)
        trainloader = DataLoader(data_train, batch_size = batch_size)
        
        training_loss_list = []
        validation_loss_list = []
        
        for e in range(epochs):
            
            batch_losses = []
            for batch in trainloader:
                batch_features: torch.Tensor = batch[0]
                batch_targets: torch.Tensor = batch[1]
                
                # reset gradient optimizer
                optimizer.zero_grad()
                
                # with the batch features, predict the batch targets
                output = model(batch_features)
                
                # compute the loss and .backward() computes the gradient of the loss function
                loss = criterion(output, batch_targets)
                loss.backward()
                
                batch_losses.append(loss.item())
                
                # update parameters (something like: params += -learning_rate * gradient)
                optimizer.step()
                
            training_loss = np.mean(batch_losses)
            validation_loss = evaluation_model_torch(model, data_validate, criterion)
            
            print(f"Fold {i} - Epoch {e+1} | training_loss: {training_loss} | validation_loss: {validation_loss}")

            training_loss_list.append(training_loss)
            validation_loss_list.append(validation_loss)
        res_fold_train[i] = training_loss_list
        res_fold_test[i] = validation_loss_list
    # for key in res_fold_train:
    #     plt.plot(res_fold_train[key], label = str(key))
    # plt.legend()
    # plt.show()
    # for key in res_fold_test:
    #     plt.plot(res_fold_test[key], label = str(key))
    # plt.legend()
    # plt.show()


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
    y_pred = X_test.copy()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    import matplotlib.pyplot as plt
    plt.hist(np.array(y_test).reshape(-1) - np.array(y_pred).reshape(-1))
    plt.show()
    plt.scatter(np.array(y_pred).reshape(-1), np.array(y_test).reshape(-1))
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
    "rnn": fit_rnn,
}