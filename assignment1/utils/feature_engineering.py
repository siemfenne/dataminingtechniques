import pandas as pd
import numpy as np
from utils.load import load_processed_data_csv
from argparse import Namespace
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl

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
    
    # for id in df.id.unique():
    #     df_sort = df[df.id==id].sort_values(by="date")
    #     r = df_sort.date.diff()
    #     print(id)
    #     print(r.value_counts())
    
    X_simple, X_recurrent, X_baseline, y = df_to_features_and_targets(args, df)

    store_features_and_targets(X_simple, X_recurrent, X_baseline, y, path = "data/")

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
    ids = df["id"]
    df = pd.get_dummies(data=df, columns=["id"])
    df["id"] = ids
    return df

def select_important_features_by_univariate_selection(args, df):
    target_column = ["mean_mood_initial"]
    exclude_from_selection = ["date"] + [c for c in df.columns if "time_window" in c or "id" in c]
    feature_columns = [c for c in df.columns if c not in target_column and c not in exclude_from_selection]
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
    return df[list(selected_features) + exclude_from_selection + target_column]

def df_to_features_and_targets(args, df: pd.DataFrame):

    X_simple = []
    X_recurrent = []
    X_baseline = []
    y = []
    
    for id in df.id.unique():
        df_id = df[df.id == id]
        for i, row in df_id.iterrows():
            target_row = df_id.loc[i]
            target_mood = target_row["mean_mood_initial"]
            if pd.isna(target_mood):
                continue
            target_date = target_row.date
            
            # for recurrent & simple, use the same window size
            feature_date_min = target_date - pd.Timedelta(args.agg_window, "D")
            feature_date_max = target_date - pd.Timedelta(1, "D")
            
            # select rows in window and check for no missing dates
            df_in_window = df_id[(df_id.date >= feature_date_min) & (df_id.date <= feature_date_max)]
            if len(df_in_window) < args.agg_window:
                continue

            # the X_recurrent part
            df_recurrent = df_in_window.copy()
            df_recurrent = df_recurrent.reset_index(drop=True)
            df_recurrent["pos"] = (np.array(df_recurrent.index) + 1)[::-1]
            df_recurrent = df_recurrent.drop(["id", "date", "mean_mood_initial"], axis = 1)
            X_recurrent.append(df_recurrent)
            
            # the baseline part
            X_baseline.append(df_recurrent.iloc[-1]["mean_mood"])
            
            # the X_simple part
            df_simple = df_in_window.copy()
            # for c in df_simple.columns:
            #     print(c)
            columns_to_agg_for = [c for c in df_simple.columns if c not in ["date", "id", "mean_mood_initial"]]
        
            agg_dict = {}
            for column in columns_to_agg_for:
                agg_dict[column] = [np.mean, np.std]
            agg_data = np.array(df_recurrent.apply(agg_dict).values).reshape(-1)
            X_simple.append(agg_data)
                
            # store target mood (same for recurrent and simple)
            y.append(target_mood)
            
    X_simple = np.array(X_simple)
    X_baseline = np.array(X_baseline).reshape(-1)
    
    return X_simple, X_recurrent, X_baseline, y
            
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

    
def store_features_and_targets(X_simple: np.ndarray, X_recurrent: list, X_baseline: np.ndarray, y: list, filename: str = "data_features", path: str = "data/"):
    
    with open(path + 'x_recurrent.pkl', 'wb') as f:
        pkl.dump(X_recurrent, f)
    with open(path + 'x_simple.pkl', 'wb') as f:
        pkl.dump(X_simple, f)
    with open(path + 'x_baseline.pkl', 'wb') as f:
        pkl.dump(X_baseline, f)
    with open(path + 'y.pkl', 'wb') as f:
        pkl.dump(y, f)
    
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