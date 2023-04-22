import pandas as pd
import pickle as pkl


def load_raw_data_csv(path="data/", filename="dataset_mood_smartphone"):
    """ Load raw data from csv """
    df = pd.read_csv(path + filename + ".csv")
    df = df.drop(df.columns[0], axis=1)
    df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)
    df['date'] = df['time'].dt.date
    return df

def load_processed_data_csv(path="data/", filename="dataset_processed"):
    """ Load the processed data (not aggregated yet) """
    df = pd.read_csv(path + filename + ".csv")
    df = df.drop(df.columns[0], axis=1)
    df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)
    return df

def load_feature_target_set(path="data/", filename="dataset_features"):
    """ Return the processed features """
    with open(path + 'x_recurrent.pkl', 'rb') as f:
        X_recurrent = pkl.load(f)
    with open(path + 'x_simple.pkl', 'rb') as f:
        X_simple = pkl.load(f)
    with open(path + 'x_baseline.pkl', 'rb') as f:
        X_baseline = pkl.load(f)
    with open(path + 'ids.pkl', 'rb') as f:
        ids = pkl.load(f)
    with open(path + 'y.pkl', 'rb') as f:
        y = pkl.load(f)
    
    return X_recurrent, X_simple, X_baseline, y, ids

