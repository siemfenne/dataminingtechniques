import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('seaborn')

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
        return predictions[:, -1]
    
    def init_state(self):
        self.hidden_cell = (torch.zeros(1,self.window_size,self.hid_dim),
                            torch.zeros(1,self.window_size,self.hid_dim))
        
def evaluation_model_torch_numpy_mse(model: nn.Module, data_validate: DataSet):
    """ Evaluate the mse of the model. mean_squared_error has some small numerical differences with torch mse """
    y_pred = model(data_validate.x_t).detach().numpy().reshape(-1,1)
    y_true = data_validate.y_t.detach().numpy().reshape(-1,1)
    return mean_squared_error(y_true, y_pred)

def cross_validate_torch(X_train, y_train, verbose = 0, lr = .001, batch_size = 10, hid_dim = 50, epochs = 10):
    n_features = X_train[0].shape[1]
    kf = KFold(10, shuffle=True, random_state=52)

    train_data = DataSet(X_train, y_train)
    fold_mse_train = []
    fold_mse_validate = []
    
    for i, (train_idx, validate_idx) in enumerate(kf.split(train_data.x_t), 1):

        # initialize model, optimizer and criterion
        model = LSTM(in_dim=n_features, hid_dim=hid_dim, out_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # split data for each fold
        data_train, data_validate = train_data.split(train_idx, validate_idx)
        
        # train on training set and validate after each epoch
        model, training_loss_list, validation_loss_list = train_torch_model(model, optimizer, criterion, data_train, data_validate, verbose, epochs = 10, batch_size=10)
        
        # store last training and validation scores
        last_training_loss, last_validation_loss = training_loss_list[-1], validation_loss_list[-1]
        fold_mse_train.append(last_training_loss)
        fold_mse_validate.append(last_validation_loss)
        
    # return the average training/validation score for all folds
    return np.mean(fold_mse_train), np.mean(fold_mse_validate)
        
    
def train_torch_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion,
    data_train: DataSet,
    data_validate: DataSet,
    verbose: int = 0,
    epochs: int = 10,
    batch_size: int = 10
):
    trainloader = DataLoader(data_train, batch_size = batch_size, shuffle = True)
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
        validation_loss = evaluation_model_torch_numpy_mse(model, data_validate)
        
        if verbose > 0:
            print(f"Epoch {e+1} | training_loss: {training_loss} | validation_loss: {validation_loss}")

        training_loss_list.append(training_loss)
        validation_loss_list.append(validation_loss)

    return model, training_loss_list, validation_loss_list