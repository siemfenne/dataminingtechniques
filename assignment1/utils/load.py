import pandas as pd


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
    X = pd.read_csv(path + filename + "_X.csv")
    y = pd.read_csv(path + filename + "_y.csv")
    return X, y