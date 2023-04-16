import pandas as pd
import numpy as np
from utils.load import load_raw_data_csv
from sklearn.preprocessing import MinMaxScaler

def clean_data():
    """ roept alle functies hieronder aan """
    
    # clean data
    df = load_raw_data_csv()
    remove_outliers(df)
    normalize_data(df)
    
    # aggregate here(?)
    
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
    t = pd.DataFrame({data.columns[0]: corrected})
    t.index = data.index
    return t
    # return corrected
    
def remove_outliers(df: pd.DataFrame):
    """
    returns dataframe for which outliers are replaced with NaN for each element in
    the variablelist using the iqr method. Values of the variablelist are first 
    transformed to log values to counter the skewness of the distributions.
    """
    columns_to_remove_outliers = [column for column in df.variable.unique() if "appCat." in column or column in ["screen", "activity"]]
    
    for column in columns_to_remove_outliers:
        
        df_var = pd.DataFrame(columns=['value'])
        df_var['value'] = np.log(df[df.variable == column].value)
        corrected = iqr_correction(data=df_var)
        df[df.variable == column].value = corrected #np.exp(corrected)
        
    return df
    
def handle_nan_values(df: pd.DataFrame):
    """ Per dag, ontbrekende waarden aanvullen met dag-gemiddelde """
    columns_to_handle_nan = ("circumplex.arousal", "circumplex.valence")

    ids = tuple(df.id.unique())
    for id in ids:
        for column in columns_to_handle_nan:
            nan_indices = df[(df.variable == column) & (df.id == id) & (df.value.isna() == True)].index
            for nan_index in nan_indices:
                nan_datum = df[nan_index].date
                df[nan_index] = np.nanmean(df[(df.date == nan_datum) & (df.id == id) & (df.variable == column)].value.values)

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