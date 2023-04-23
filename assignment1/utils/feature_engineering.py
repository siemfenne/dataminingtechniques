import pandas as pd
import numpy as np
from utils.load import load_processed_data_csv
from argparse import Namespace
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
from tqdm import tqdm

def feature_engineering(args: Namespace):
    # load cleaned data
    df = load_processed_data_csv()
    
    if args.window > 0:
        df = add_categorical_for_part_of_day(df, args.window)
    else:
        print("Creating pivot table, not adding part-of-day aggregation")
        # aggregate also full days, using both sum and mean
        # all screen time related values use sum, else mean
        df = df.pivot_table(values='value', index=['id', 'date'],
                                columns=['variable'], aggfunc=[np.sum, np.mean], fill_value=np.nan)
        for column in df["sum"].columns:
            if column in ["mood", "circumplex.arousal", "circumplex.valence"]:
                del df["sum", column]
            else:
                del df["mean", column]
        df.columns = [c[0] + "_" + c[1] for c in df.columns]
    
    df.reset_index()
    df = missing_value_interpolation(df)
    df = userid_to_dummies(df)
    df = add_weekday_dummy(df)
    df = select_important_features_by_univariate_selection(args, df)
    
    X_simple, X_recurrent, X_baseline, y, ids = df_to_features_and_targets(args, df)
    
    normalize_features = True
    if normalize_features:
        print("Normalizing features (only recurrent and simple) ...")
        # normalize non-temporal data
        X_simple = normalize_data(X_simple)
        
        # normalize temporal data (concat -> normalize -> transform back to seperate sequences)
        n_sequences = len(X_recurrent)
        X_recurrent = normalize_data(pd.concat(X_recurrent))
        X_recurrent = [X_recurrent[i:i+args.agg_window].reset_index(drop=True) for i in range(n_sequences)]
        
        # baseline not reshaped (lagged value == prediction)
        
    store_features_and_targets(X_simple, X_recurrent, X_baseline, y, ids, path = "data/")

def add_categorical_for_part_of_day(df: pd.DataFrame, window: int = 6):
    """ 
    Under the hypothesis the time of day of user activity can be predictive for mood,
    the count and sum is computed for the recorded variables related to screen time
    """
    
    if 24 % window > 0:
        raise ValueError("24 Must be a multiple of the window")
    
    # create a new column indicating the time window for each datetime value
    periods = int(24/window)
    bins = [i*window for i in range(periods+1)]
    labels = [f"{i*window}-{(1+i)*window}" for i in range(periods)]
    df["time_window"] = pd.cut(
        df['time'].dt.hour,
        bins=bins, 
        labels=labels,
        include_lowest=True
    )
        
    # aggregate part of day to days, using count and sum for all screen variables
    # because all variables related to screen time, nan values automatically to 0!
    df_time_window = df[(df.variable != "mood") & (df.variable != "circumplex.arousal") & (df.variable != "circumplex.valence")]
    df_time_window = df_time_window \
        .pivot_table(values=["value"], index=["id", "date", "time_window"], columns=["variable"], aggfunc=["count", np.sum], fill_value=0) \
        .reset_index().pivot(index=["id", "date"], columns=["time_window"])
    df_time_window.columns = [f"time_window_{c[0]}_{c[2]}_{c[3]}" for c in df_time_window.columns]
    for time_label in labels:
        del df_time_window["time_window_sum_sms_" + time_label]
        del df_time_window["time_window_sum_call_" + time_label]

    # aggregate also full days, using both sum and mean
    # all screen time related values use sum, else mean
    df_attr = df.pivot_table(values='value', index=['id', 'date'],
                            columns=['variable'], aggfunc=[np.sum, np.mean], fill_value=np.nan)
    for column in df_attr["sum"].columns:
        if column in ["mood", "circumplex.arousal", "circumplex.valence"]:
            del df_attr["sum", column]
        else:
            del df_attr["mean", column]
    df_attr.columns = [c[0] + "_" + c[1] for c in df_attr.columns]
     
    # merge the 2 dataframes together
    return pd.merge(df_attr, df_time_window, on=["date", "id"], how="left")
    
def missing_value_interpolation(df: pd.DataFrame):
    """ add a categorial variable which indicates the part of day """
    # for all variables containing sum (i.e. related to screen time), nan values are set to zero
    columns_to_fill_zero = [c for c in df.columns if "sum_" in c or "count_" in c]
    df[columns_to_fill_zero] = df[columns_to_fill_zero].fillna(0)
    columns_to_impute_otherwise = set([c for c in df.columns if "sum_" in c or "mean_" in c]) - set(columns_to_fill_zero)
    print(f"Imputing missing values with 0, except for columns: {columns_to_impute_otherwise}")
    print(f"Imputing missing values through KNN for columns: {columns_to_impute_otherwise}")
    
    # store copy of moods so targets are not interpolated
    moods = df.copy()["mean_mood"]
    
    # important for KNN neighbors regression
    df_norm = normalize_data(df.copy())
    
    df = df.reset_index() # add two columns with id and date from index
    df.date = pd.to_datetime(df.date)
    df_norm = df.reset_index()
    df_norm.date = pd.to_datetime(df_norm.date)
    
    # select rows where target is known
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import KFold, GridSearchCV
    
    def own_scorer(mod, features, targets):
        """ 
        The scoring metric for the KNN gridsearch.
        GridSearchCV maximizes the loss function, therefore this function returns negative MSE
        """
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
            param_grid = {"n_neighbors": [2, 4, 8, 16, 32, 48], "weights": ["uniform", "distance"]},
            scoring = own_scorer,
            error_score="raise"
        )
        mod.fit(train[features].values, train[column].values)
        print(f"KNN result: {column} - best_params: {mod.best_params_} - best_score: {mod.best_score_}")
        pred = mod.predict(test[features])
        df.loc[pd.isna(df[column]), column] = pred
    
    df["mean_mood_initial"] = moods.values
    return df

def userid_to_dummies(df: pd.DataFrame):
    """ Drops the user id column and adds user id dummies to the dataframe """
    print("User id column to dummies ...")
    ids = df["id"]
    df = pd.get_dummies(data=df, columns=["id"])
    df["id"] = ids
    return df

def select_important_features_by_univariate_selection(args, df):
    """ 
    For a list of columns (exclude the original target column + some other columns), perform F regression and select the most important features.
    These features are used later on in the temporal data and/or used for further aggregation
    """
    target_column = ["mean_mood_initial"]
    # exclude_from_selection = ["date"] + [c for c in df.columns if "time_window" in c or "id" == c] # exclude id itself, include dummies in SelectKBest
    exclude_from_selection = ["date", "id", "is_weekend"]
    feature_columns = [c for c in df.columns if c not in target_column and c not in exclude_from_selection]
    features = df[feature_columns]
    
    print("Selecting most important features out of:\n", feature_columns)
    targets = df["mean_mood_initial"].shift(-1)
    features = features[targets.notna()]
    targets = targets.dropna()
    from sklearn.feature_selection import SelectPercentile, SelectKBest, f_regression
    s = SelectKBest(f_regression, k = args.k_features)
    s.fit(features, targets)
    selected_features = features.columns[s.get_support(indices=True)]
    print(f"Best {args.k_features} features are:\n", selected_features)
    
    # return selected features + targets and date
    return df[list(selected_features) + exclude_from_selection + target_column]

def df_to_features_and_targets(args, df: pd.DataFrame):
    """ 
    For each user, we iterate through the date-ordered data for each user:
    - for each data point (the target), we also select the desired feature window from (t-1 to t-{window_size})
    - check if there are no missing values present, i.e. no date gaps
    - drop date columns (unsuitable as features for the model) and some other irrelevant columns
    - store these temporal features 
    - aggregate these temporal features to make suitable for lightGBM for example, by
        - mean, std, min, max, etc.
    - store this aggregated result as well
    - the baseline features are the avg mood at t-1
    - return temporal, simple, baseline features + the target (mean mood of today)
    """
    X_simple = []
    X_recurrent = []
    X_baseline = []
    ids = []
    y = []

    print("Transforming data to (temporal) features and targets ...")
    for id in tqdm(df.id.unique()):
        df_id = df[df.id == id]
        
        for i, row in df_id.iterrows():
            target_row = df_id.loc[i]
            target_mood = target_row["mean_mood_initial"]
            target_is_weekend = target_row["is_weekend"]
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
            df_simple = df_in_window.copy().reset_index(drop=True)
            df_simple = df_simple.drop(["id", "date", "mean_mood_initial", "is_weekend"], axis = 1)
            
            columns_to_agg_for_continuous = [c for c in df_simple.columns if "id_" not in c]
            columns_to_store_last = list(set([c for c in df_simple.columns]) - set(columns_to_agg_for_continuous))

            def wm(s: pd.Series):
                s = s.values
                return np.average(s, weights = [i+1 for i in range(len(s))])
        
            def select_first(s: pd.Series):
                return s.values[0]
        
            agg_dict = {}
            for column in columns_to_agg_for_continuous:
                agg_dict[column] = [np.mean, np.std]#[np.mean, np.std, np.max, np.min, wm]
            for column in columns_to_store_last:
                agg_dict[column] = [select_first]
                
            # apply the aggregation dict and convert the data to a single row
            df_agg = df_simple \
                .apply(agg_dict).reset_index() \
                .pivot(columns="index").sum(skipna=True, min_count=1).dropna()
            df_agg["is_weekend_target"] = target_is_weekend
            agg_array = np.array(df_agg.values).reshape(-1)
            X_simple_columns = [c[0] + "_" + c[1] for c in df_agg.index]
            
            # store simple data
            X_simple.append(agg_array)
                
            # store target mood (same for recurrent and simple)
            y.append(target_mood)
            
            # store id, used for stratified split later on
            ids.append(id)
            
    # print the final column feature for simple
    print("The final features for recurrent:\n", X_recurrent[0].columns)
    print("The final features for simple:\n", X_simple_columns)
    
    X_simple = pd.DataFrame(columns = X_simple_columns, data = np.array(X_simple))
    # X_simple = np.array(X_simple)
    X_baseline = np.array(X_baseline).reshape(-1)
    
    return X_simple, X_recurrent, X_baseline, y, ids

def normalize_data(df: pd.DataFrame):
    """ Normalize the features """
    columns_to_normalize = set(df.columns) - set([c for c in df.columns if "id_" in c]) - set(["date", "id", "mean_mood_initial"])
    # columns_to_normalize = [column for column in df.variable.unique() if "appCat." in column or column in ["screen", "activity"]]
    scaler = MinMaxScaler()
    
    for column in columns_to_normalize:
        values = df[column]
        if len(values) > 0:
            df[column] = scaler.fit_transform(values.values.reshape(-1,1))
    return df

def add_weekday_dummy(df: pd.DataFrame):
    """ Add a dummy, indicating whether or not the current day is a Saturday/Sunday"""
    week_ints = df["date"].dt.weekday
    df["is_weekend"] = np.where(week_ints >= 5 , 1, 0)
    return df
    
def store_features_and_targets(X_simple: np.ndarray, X_recurrent: list, X_baseline: np.ndarray, y: list, ids: list, filename: str = "data_features", path: str = "data/"):
    
    with open(path + 'x_recurrent.pkl', 'wb') as f:
        pkl.dump(X_recurrent, f)
    with open(path + 'x_simple.pkl', 'wb') as f:
        pkl.dump(X_simple, f)
    with open(path + 'x_baseline.pkl', 'wb') as f:
        pkl.dump(X_baseline, f)
    with open(path + 'ids.pkl', 'wb') as f:
        pkl.dump(ids, f)
    with open(path + 'y.pkl', 'wb') as f:
        pkl.dump(y, f)
        
def check_for_nan(df: pd.DataFrame):
    """ A function to easily check for all nan values present in the dataframe """
    for column in df.columns:
        nan_count = np.sum(df[column].isna())
        if nan_count > 0:
            print(f"column: {column} - nan_count: {nan_count} - total val. present: {len(df[column])}")