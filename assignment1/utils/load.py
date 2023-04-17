import pandas as pd
import pickle as pkl


def load_raw_data_csv(path="data/", filename="dataset_mood_smartphone"):
    """ Load data from csv and pivot table blabla """
    df = pd.read_csv(path + filename + ".csv")
    df = df.drop(df.columns[0], axis=1)
    df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)
    df['date'] = df['time'].dt.date
    # df = pd.pivot_table(df, values='value', index=['id', 'date'],
                                    # columns=['variable'], aggfunc=[np.sum, np.mean]).reset_index()
    return df

def load_processed_data_csv(path="data/", filename="dataset_processed"):
    """ Load the processed data """
    df = pd.read_csv(path + filename + ".csv")
    df = df.drop(df.columns[0], axis=1)
    df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)
    # df = pd.pivot_table(df, values='value', index=['id', 'date'],
                                    # columns=['variable'], aggfunc=[np.sum, np.mean]).reset_index()
    return df

def load_feature_target_set(path="data/", filename="dataset_features"):
    """ return the processed features """
    with open(path + 'x_recurrent.pkl', 'rb') as f:
        X_recurrent = pkl.load(f)
    with open(path + 'x_simple.pkl', 'rb') as f:
        X_simple = pkl.load(f)
    with open(path + 'x_baseline.pkl', 'rb') as f:
        X_baseline = pkl.load(f)
    with open(path + 'y.pkl', 'rb') as f:
        y = pkl.load(f)
        
    # import numpy as np
    # train_pct = .8
    # train_index = np.random.choice(range(len(y)), replace=False, size = int(train_pct*len(y)))
    # test_index = 
    
    return X_recurrent, X_simple, X_baseline, y

