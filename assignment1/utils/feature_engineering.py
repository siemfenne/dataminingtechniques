import pandas as pd
import numpy as np
from utils.load import load_processed_data_csv

def feature_engineering():
    df = load_processed_data_csv()
    df = pd.pivot_table(df, 
        values='value', 
        index=['id', 'date'],
        columns=['variable'], 
        aggfunc=[np.sum, np.mean, "count"]).reset_index()
    
    # remove periods with inconsecutive days of mood data
    df = remove_days_with_no_mood(df)
    
    # compute the features
    df = compute_features(df)
    
    # dropna values from targets and store
    targets = df.mood.shift(-1)
    targets = targets.dropna()
    features = features[targets.index]
    store_features_and_targets(df)
    
def remove_days_with_no_mood(df):
    return df[df.mood.count > 0]
    
def find_consecutive_moods(df, window = 5):
    start_end_indices = []
    return start_end_indices

def compute_features(df, window = 5):
    features_to_keep = df.columns
    # indices = find_consecutive_moods()
    targets = df["mood"]
    features = df.rolling(window).apply(agg_config)
    
def store_features_and_targets(features: pd.DataFrame, targets: pd.DataFrame, filename: str = "data_features", path: str = "data/"):
    features.to_csv(path + filename + "_X.csv")
    targets.to_csv(path + filename + "_y.csv")
    
agg_config = {
    "appCat.builtin": [],
    "appCat.communication": [],
    "appCat.entertainment": [],
    "appCat.finance": [],
    "appCat.game": [],
    "appCat.office": [],
    "appCat.other": [],
    "appCat.social": [],
    "appCat.travel": [],
    "appCat.unknown": [],
    "appCat.utilities": [],
    "appCat.weather": [],
    "appCat.": [],
    "appCat.": [],
    "appCat.": [],
    "appCat.": [],
    "screen": [],
    "call": [],
    "sms": [],
    "activity": [],
    "circumplex.arousal": [],
    "circumplex.valence": [],
    "mood": [],
}