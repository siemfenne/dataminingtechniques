import pandas as pd
import numpy as np
from utils.load import load_processed_data_csv
from argparse import Namespace
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

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
    df = missing_value_interpolation(df)
    df = userid_to_dummies(df)
    df = select_important_features_by_univariate_selection(args, df)
    
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.heatmap(df[df.columns[:25]].corr())
    # plt.show()
    # v = df.corr()["mean_mood_initial"].sort_values()
    # print(v)
    # features, targets = df_to_features_and_targets(df, args)
    
    # store_features_and_targets(features, targets, path = "data/")
    
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
    # for all variables containing sum (i.e. related to screen time), nan values are set to zero
    columns_to_fill_zero = [c for c in df.columns if "sum_" in c]
    df[columns_to_fill_zero] = df[columns_to_fill_zero].fillna(0)
    columns_to_impute_otherwise = set([c for c in df.columns if "sum_" in c or "mean_" in c]) - set(columns_to_fill_zero)
    
    # store copy of moods so targets are not interpolated
    moods = df.copy()["mean_mood"]
    
    df_norm = normalize_data(df.copy())
    
    df = df.reset_index() # add two columns with id and date
    df.date = pd.to_datetime(df.date)
    df_norm = df.reset_index()
    df_norm.date = pd.to_datetime(df_norm.date)
    
    # select rows where target is known
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import KFold, GridSearchCV
    
    def own_scorer(mod, features, targets):
        pred = mod.predict(features)
        # return -mean_absolute_percentage_error(targets, pred)
        return -mean_squared_error(targets, pred)
    
    for column in columns_to_impute_otherwise:
        train = df_norm[df_norm[column].notna()]
        test = df_norm[df_norm[column].isna()]
        
        features = set(df.columns) - set(list(columns_to_impute_otherwise) + ["date", "id"])
          
        mod = GridSearchCV(
            estimator = KNeighborsRegressor(),
            cv = KFold(10, shuffle=True, random_state=42),
            param_grid = {"n_neighbors": [7,12, 15, 20, 25, 30], "weights": ["uniform", "distance"]},
            scoring = own_scorer,
            error_score="raise"
        )
        mod.fit(train[features].values, train[column].values)
        print(column, mod.best_params_, mod.best_score_)
        pred = mod.predict(test[features])
        df.loc[pd.isna(df[column]), column] = pred
    
    df["mean_mood_initial"] = moods.values
    return df

def userid_to_dummies(df: pd.DataFrame):
    df = pd.get_dummies(data=df, columns=["id"])
    return df

def select_important_features_by_univariate_selection(args, df):
    targets = ["mean_mood_initial"]
    exclude_from_selection = ["date"] + [c for c in df.columns if "time_window" in c or "id_" in c]
    feature_columns = [c for c in df.columns if c not in targets and c not in exclude_from_selection]
    features = df[feature_columns]
    print("Selecting features on: \n", feature_columns)
    
    targets = df["mean_mood_initial"].shift(-1)
    features = features[targets.notna()]
    targets = targets.dropna()
    from sklearn.feature_selection import SelectPercentile, SelectKBest, f_regression
    s = SelectKBest(f_regression, k = args.k_features)
    res = s.fit_transform(features, targets)
    selected_features = features.columns[s.get_support(indices=True)]
    print("Selected features are: \n", selected_features)
    
    # return selected features + targets and date
    return df[list(selected_features) + exclude_from_selection + targets]

def df_to_features_and_targets(df: pd.DataFrame):
    
    return pd.DataFrame(), pd.DataFrame()
            
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

def normalize_data(df: pd.DataFrame):
    columns_to_normalize = set(df.columns) - set(["mean_mood", "mean_circumplex.arousal", "mean_circumplex.valence"])
    # columns_to_normalize = [column for column in df.variable.unique() if "appCat." in column or column in ["screen", "activity"]]
    scaler = MinMaxScaler()
    
    for column in columns_to_normalize:
        values = df[column]
        if len(values) > 0:
            df[column] = scaler.fit_transform(values.values.reshape(-1,1))
    return df

    
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