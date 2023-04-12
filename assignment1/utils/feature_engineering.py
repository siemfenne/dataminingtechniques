import pandas as pd
import numpy as np
from load import load_processed_data_csv

def feature_engineering():
    df = load_processed_data_csv()
    df = pd.pivot_table(df, 
        values='value', 
        index=['id', 'date'],
        columns=['variable'], 
        aggfunc=[np.sum, np.mean]).reset_index()

    store_features_and_targets(df)
    
def find_consecutive_moods(df):
    start_end_indices = []
    return start_end_indices
    
def store_features_and_targets(df: pd.DataFrame, filename: str = "data_features_targets", path: str = "../data/"):
    df.to_csv(path + filename + ".csv")