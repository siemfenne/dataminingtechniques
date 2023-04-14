from data_cleaning import clean_data
from feature_engineering import feature_engineering
from model_estimation import model_functions

if __name__ == "__main__":
    clean_data()
    feature_engineering()
    model_functions["xgb"]()