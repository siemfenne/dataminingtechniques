import pandas as pd
import numpy as np
from utils.load import load_raw_data_csv
from sklearn.preprocessing import MinMaxScaler

def clean_data():
    """ roept alle functies hieronder aan """
    
    # clean data
    print("Loading raw data from csv")
    df = load_raw_data_csv()
    print("Removing outliers to nan ...")
    df = remove_outliers(df)
    
    print("Setting nan values to daily mean for same userID ...")
    df = handle_nan_values(df)
    
    # for c in df.variable.unique():
    #     val = df[df.variable == c].value
    #     nan_count = val.isna().sum()
    #     nan_rel = nan_count / len(val)
    #     print(c, nan_count, nan_rel)
        
    # print("Normalizing data")
    # normalize_data(df)

    store_processed_data(df)

def iqr_correction(data: pd.DataFrame):
    """
    Applies iqr outlier detection method to dataframe and 
    replaces the outliers with NaN values
    """

    Q1 = data.value.quantile(0.25)
    Q3 = data.value.quantile(0.75)
    IQR = Q3 - Q1
    lim_width = 1.5
    lower_lim = Q1 - (lim_width*IQR)
    upper_lim = Q3 + (lim_width*IQR)
    corrected = np.where(data.value > upper_lim, np.nan, np.where(data.value < lower_lim, np.nan, data.value))
    data["value"] = corrected
    return data
    
def remove_outliers(df: pd.DataFrame):
    """
    returns dataframe for which outliers are replaced with NaN for each element in
    the variablelist using the iqr method. Values of the variablelist are first 
    transformed to log values to counter the skewness of the distributions.
    """
    # klopt deze lijst
    columns_to_remove_outliers = [column for column in df.variable.unique() if "appCat." in column or column in ["screen"]]
    columns_to_not_remove_outliers = set(df.variable.unique()) - set(columns_to_remove_outliers)
    
    data_frames = []
    for column in columns_to_remove_outliers:
        # outlier for
        df_var = df[df.variable == column]
        df_var["value"] = np.log(df_var.value)
        data_frames.append(iqr_correction(df_var))
        
    for column in columns_to_not_remove_outliers:
        data_frames.append(df[df.variable == column])
        
    return pd.concat(data_frames)
    
def handle_nan_values(df: pd.DataFrame):
    """ Per dag, ontbrekende waarden aanvullen met dag-gemiddelde """
    # the only columns with nan values in the raw data
    # klopt deze lijst (activity, circumplex)
    columns_to_handle_nan = [c for c in df.variable.unique() if "appCat." in c or c in ["screen", "circumplex.arousal", "circumplex.valence"]]

    ids = tuple(df.id.unique())
    
    for column in columns_to_handle_nan:
        uncond_mean_var = np.nanmean(df[df.variable == column].value)
        
        for id in ids:
            df_id_var = df[(df.variable == column) & (df.id == id)]
            uncond_mean_id_var = np.nanmean(df_id_var.value)
            nan_indices = df_id_var[df_id_var.value.isna() == True].index
            
            for nan_index in nan_indices:
                nan_datum = df_id_var.loc[nan_index].date
                
                # the mean of the other circumplex.. values for day+userId
                daily_mean = np.nanmean(df_id_var[(df_id_var.date == nan_datum)].value.values)

                if pd.notna(daily_mean):
                    df.loc[nan_index, "value"] = daily_mean
                elif pd.notna(uncond_mean_id_var):
                    df.loc[nan_index, "value"] = uncond_mean_id_var
                elif pd.notna(uncond_mean_var):
                    df.loc[nan_index, "value"] = uncond_mean_var
                else:
                    raise Exception(f"Could not impute nan values for: {column} - {id} - {nan_datum} with index {nan_index}")
    return df

def normalize_data(df: pd.DataFrame):
    columns_to_normalize = [column for column in df.variable.unique() if "appCat." in column or column in ["screen", "activity"]]
    scaler = MinMaxScaler()
    ids = tuple(df.id.unique())
    
    for column in columns_to_normalize:
        values = df[df.variable == column].value
        if len(values) > 0:
            df[df.variable == column].value = scaler.fit_transform(values.values.reshape(-1,1))
    return df

def store_processed_data(df: pd.DataFrame, filename: str = "dataset_processed", path: str = "data/"):
    total_file_path = path + filename + ".csv"
    print("Storing file under path %s ..." % (total_file_path))
    df.to_csv(total_file_path)