import argparse

from utils.data_cleaning import clean_data
from utils.feature_engineering import feature_engineering
from utils.model_estimation import model_functions

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--clean", help="true if clean data", action = "store_true", default=True)
    parser.add_argument("-f", "--feature", help="true if feature engineering", action = "store_true", default=True)
    parser.add_argument("-m", "--models", nargs="+", help="models to train, can be: " + " ".join(list(model_functions.keys())), action="append", default=["xgb"])
    
    args = parser.parse_args()
    args.models = args.models[0]
    
    # redo cleaning and/or feature engineering
    if args.clean:
        clean_data()
    if args.clean or args.feature:
        feature_engineering()
        
    # cross validate, train and evaluate out-of-sample each model
    for model in parser.models:
        model_functions[model]()