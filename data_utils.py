import pandas as pd
import numpy as np


def process_data(filepath: str = "./", filename: str = "dataset_mood_smartphone"):
    df = pd.read_csv(filename + ".csv")
    
    
    
def preprocess_data(df: pd.DataFrame, filepath: str, filename: str):
    store_processed_data_at = filepath + filename + ".csv"
    
    
    df.to_csv()
    