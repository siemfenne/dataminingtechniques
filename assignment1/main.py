import argparse
from utils.data_cleaning import clean_data
from utils.feature_engineering import feature_engineering
from utils.model_estimation import model_functions
from utils.functions import print_section
from dieboldmariano import dm_test
import warnings
warnings.simplefilter('ignore')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--criterium", help="the selection criterium (lgb or kbest)", type=str, default="kbest")
    parser.add_argument("-s", "--steps", help="the steps to perform c(lean)-f(eature)-m(odels)", type=str, default="cfm")
    parser.add_argument("-f", "--feature", help="true if feature engineering", const=bool, default=False, action="store_const")
    parser.add_argument("-m", "--models", nargs="+", help="models to train, can be: " + " ".join(list(model_functions.keys())), action="store", default=["baseline", "lgb", "rnn"])
    parser.add_argument("-w", "--window", help="count for time window", type=int, default=6)
    parser.add_argument("-k", "--k_features", help="the number of features to select", type=int, default=20)
    parser.add_argument("-a", "--agg_window", help="the window (of days) to aggregate on", type=int, default=5)
    args = parser.parse_args()
    
    # redo cleaning and/or feature engineering
    if "c" in args.steps:
        print_section("CLEANING DATA")
        clean_data()
    if "f" in args.steps:
        print_section("FEATURE ENGINEERING")
        feature_engineering(args)
    if "m" in args.steps:
        print_section("TRAINING AND EVALUATING MODELS")
        
        # store the predictions + true values for each model
        res = {}
        for model in args.models:
            y_pred, y_test = model_functions[model](args)
            res[model] = [y_pred, y_test]
        
        # perform diebold mariano test for all models compared to the baseline
        # only possible if baseline included
        if "baseline" in args.models:
            y_pred_baseline, y_true = res["baseline"]
            for model in args.models:
                if model != "baseline":
                    y_pred, _ = res[model]
                    stat, p = dm_test(y_true, y_pred, y_pred_baseline, one_sided=True, )
                    print(f"Diebold Mariano - {model.upper()} vs. baseline - stat: {stat} - p_val: {p}")