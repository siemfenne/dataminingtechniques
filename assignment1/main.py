import argparse
from utils.data_cleaning import clean_data
from utils.feature_engineering import feature_engineering
from utils.model_estimation import model_functions

import warnings
warnings.simplefilter('ignore')

if __name__ == "__main__":
    
    # werkt niet, hoeren ding
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--clean", help="true if clean data", type=bool, default=False)
    parser.add_argument("-f", "--feature", help="true if feature engineering", type=bool, default=False)
    parser.add_argument("-m", "--models", nargs="+", help="models to train, can be: " + " ".join(list(model_functions.keys())), action="append", default=[["rnn"]])
    parser.add_argument("-w", "--window", help="count for time window", type=int, default=6)
    parser.add_argument("-k", "--k_features", help="the number of features to select", type=int, default=20)
    parser.add_argument("-a", "--agg_window", help="the window (of days) to aggregate on", type=int, default=5)
    
    args = parser.parse_args()
    args.models = args.models[0]
    
    # redo cleaning and/or feature engineering
    if args.clean:
        clean_data()
    if args.clean or args.feature:
        feature_engineering(args)
        
    # cross validate, train and evaluate out-of-sample each model
    for model in args.models:
        model_functions[model]()
