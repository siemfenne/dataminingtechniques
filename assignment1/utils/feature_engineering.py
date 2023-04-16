import pandas as pd
import numpy as np
from utils.load import load_processed_data_csv
from argparse import Namespace

def feature_engineering(args: Namespace):
    # load cleaned data
    df = load_processed_data_csv()
    
    if args.window:
        df = add_categorical_for_part_of_day(df, args.window)
    else:
        # pivot table
        df = pd.pivot_table(df, 
            values='value',
            index=['id', 'date'],
            columns=['variable'],
            aggfunc=[np.sum, np.mean, "count"]).reset_index()
        for column in df["sum"].columns:
            if column in ["mood", "circumplex.arousal", "circumplex.valence"]:
                del df["sum", column]
            else:
                del df["mean", column]
        df.columns = [c[0] + "_" + c[1] for c in df.columns]
    
    df.reset_index()
    missing_value_interpolation(df)
    print(df)
    features, targets = aggregate_time_windows(df, args)
    
    store_features_and_targets(features, targets, path = "data/")
    
def add_categorical_for_part_of_day(df: pd.DataFrame, window: int = 6):
    
    if 24 % window > 0:
        raise Exception("window not allowed")
    
    periods = int(24/window)
    bins = [i*window for i in range(periods+1)]
    labels = [f"{i*window}-{(1+i)*window}" for i in range(periods)]

    # create a new column indicating the time window for each datetime value
    df["time_window"] = pd.cut(
        df['time'].dt.hour,
        bins=bins, 
        labels=labels,
        include_lowest=True
    )
    df = pd.get_dummies(df, columns=["time_window"])
    # del df["time_window"]
    
    # aggregate part of day to days, using count
    df_time_window = pd.pivot_table(df, values=["time_window_" + l for l in labels], index=['id', 'date'],
                            columns=['variable'], aggfunc=['count'], fill_value=0)

    # aggregate attributes to days, using both sum and mean
    df_attr = pd.pivot_table(df, values='value', index=['id', 'date'],
                            columns=['variable'], aggfunc=[np.sum, np.mean], fill_value=np.nan)
    
    for column in df_attr["sum"].columns:
        if column in ["mood", "circumplex.arousal", "circumplex.valence"]:
            del df_attr["sum", column]
        else:
            del df_attr["mean", column]
    
    df_attr.columns = [c[0] + "_" + c[1] for c in df_attr.columns]
    # df_attr.rename(columns="_".join, inplace=True)
    df_time_window.columns = list(df_time_window["count"].columns)
    df_time_window.rename(columns="_".join, inplace=True)
    df = pd.merge(df_attr, df_time_window, on=["date", "id"])
    return df
    
def missing_value_interpolation(df: pd.DataFrame):
    """ add a categorial variable which indicates the part of day """
    # for all screen times, fill in zero
    columns_to_fill_zero = [c for c in df.columns if "sum_" in c]
    df[columns_to_fill_zero] = df[columns_to_fill_zero].fillna(0)
    
    # circumplex and mood
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt
    var = "circumplex.arousal"
    d = df[f"mean_{var}"].loc[:, "AS14.01"]
    # plot_pacf(d.dropna().iloc[2:])
    # plt.show()
    
def remove_days_with_no_mood(df):
    return df[df.mood.count > 0]
    
def find_consecutive_moods(df, window = 5):
    start_end_indices = []
    return start_end_indices
    
def aggregate_time_windows(df, args):
    
    features = pd.DataFrame()
    targets = pd.DataFrame()
    
    for i in range(args.window, len(df)-1):
        if df.loc[i]["mean_mood"].isna().value:
            continue
        

    
    return features, targets
    
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
    "screen": [],
    "call": [],
    "sms": [],
    "activity": [],
    "circumplex.arousal": [],
    "circumplex.valence": [],
    "mood": [],
}