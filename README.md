# Data Mining Techniques VU Assignment 1
## Basic Description
This model analyses user mobile activity data. The data contains activity logs from several users, like screen time (per specific app category), dummy variables indicating a phone call, etc.. It also contains self reported mood scores. The aim of this project is to predict the average reported mood of a user one day ahead, based on this activity. The project is seperated in three parts.
- Cleaning the raw data. Think of outlier corrections and imputing missing values.
- Feature engineering. To make the data suitable for regression, it is aggregated on by days (and part of days). Not all variables are present, therefore again missing values are imputed. By F-regression or LightGBM, the most likely predictive features are selected. A 5 day window is selected for temporal data. For non-temporal models, the data is aggregated over this 5-day window is aggregated.
- Model estimation: the specification of LightGBM and LSTM is determined by cross validation on the training data. Then the final evaluation is done on the unseen test set. The Diebold-Mariano test is also performed.

## Usage
To be able to run the program without errors, the original dataset_mood_smartphone.csv needs to be present under the path ./data/dataset_mood_smartphone.csv, relative to main.py.
The project can be executed by running python (or your respective python command) main.py. There are a number of functionalities build in which a user can manually adjust, just run python main.py --help. By default it will clean the data, engineer features and train the baseline, rnn and lightgbm model.
- python main.py --models baseline lgb rnn --window 6 --steps cfm --agg_window 5 --k_features 20