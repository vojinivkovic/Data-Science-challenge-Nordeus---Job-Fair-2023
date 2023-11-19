import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")


def fill_nan_values(data):
    #Fill nan values in column
    data["global_competition_level"].fillna(0, inplace=True)
    return

def encode_categorical_data(data):
    #Encoding categorical data with handmade encoding

    encode_values = {"0) NonPayer": 1,
                     "1) ExPayer": 2,
                     "2) Minnow": 3,
                     "3) Dolphin": 4,
                     "4) Whale": 5}

    for key, value in encode_values.items():
        idx = data["dynamic_payment_segment"] == key
        data.loc[idx, "dynamic_payment_segment"] = value

    return

def dynamic_payment_segment_preprocess(data):

    unique_teams_counts = np.unique(data["dynamic_payment_segment"], return_counts=True)
    unique_teams = np.flip(unique_teams_counts[0])
    counts = np.flip(unique_teams_counts[1])

    new_values = [10]

    for i in range(len(counts)-1):
        new_values.append(new_values[-1] + counts[i])

    for j in range(len(new_values)):
        idx = data["dynamic_payment_segment"] == unique_teams[j]
        data.loc[idx, "dynamic_payment_segment"] = new_values[j]

def avg_stars_top_3_bench_players(data):
    #Making new column that will be more representative than avg_stars_top_14_players

    data["avg_stars_top_14_players"] = (14*data["avg_stars_top_14_players"] - 11*data["avg_stars_top_11_players"]) / 3
    data.rename(columns={"avg_stars_top_14_players": "avg_stars_top_3_bench_players"}, inplace=True)

def digitize_last_28_days(data):

    idx_1 = (data["days_active_last_28_days"] >= 0) & (data["days_active_last_28_days"] <= 7)
    idx_2 = (data["days_active_last_28_days"] >= 8) & (data["days_active_last_28_days"] <= 14)
    idx_3 = (data["days_active_last_28_days"] >= 15) & (data["days_active_last_28_days"] <= 21)
    idx_4 = data["days_active_last_28_days"] >= 22

    data.loc[idx_1, "days_active_last_28_days"] = 1
    data.loc[idx_2, "days_active_last_28_days"] = 2
    data.loc[idx_3, "days_active_last_28_days"] = 3
    data.loc[idx_4, "days_active_last_28_days"] = 4

def add_new_feature(data):

    kmeans = KMeans(n_clusters=14)
    kmeans.fit(data)
    data["favorites"] = kmeans.labels_
    data["favorites"] = data["favorites"] + 1

def preprocess_data_for_leagues(data):
    #Preprocessing data within the same league and scaling the data
    leagues = np.unique(data["league_id"])
    avg_stars_top_3_bench_players(data)

    min_max_columns = ["cohort_season", "avg_age_top_11_players",
                       "avg_stars_top_11_players", "avg_training_factor_top_11_players",
                       "league_match_watched_count_last_28_days", "session_count_last_28_days",
                       "playtime_last_28_days", "league_match_won_count_last_28_days",
                       "training_count_last_28_days", "global_competition_level", "tokens_spent_last_28_days",
                       "tokens_stash", "rests_stash", "morale_boosters_stash",
                       "avg_stars_top_3_bench_players", "days_active_last_28_days"]


    for league in leagues:
        idx = data["league_id"] == league
        temp_data = data.loc[idx, :]
        for column in min_max_columns:
            max_value = np.max(temp_data[column])
            min_value = np.min(temp_data[column])
            if(max_value == min_value):
                temp_data[column] = 0
            else:
                temp_data[column] = (temp_data[column] - min_value) / (max_value - min_value)
        dynamic_payment_segment_preprocess(temp_data)

        data.loc[idx, :] = temp_data

    #add_new_feature(data)

def prepare_train_data(data):

    fill_nan_values(data)
    X_train = data.drop(columns=["league_rank", "season", "club_id", "registration_country", "registration_platform_specific"])
    Y_train = data["league_rank"]
    encode_categorical_data(X_train)
    preprocess_data_for_leagues(X_train)
    X_train.drop(columns="league_id", inplace=True)

    return X_train, Y_train

def prepare_test_data(data):

    fill_nan_values(data)
    X_test = data.drop(columns=["season", "club_id", "registration_country", "registration_platform_specific"])
    encode_categorical_data(X_test)
    preprocess_data_for_leagues(X_test)
    X_test.drop(columns="league_id", inplace=True)

    return X_test

def make_synthetic_data(X_data, Y_data):
    #Make synthetic data using SMOTE oversampling technique and making artificial unbalanced data

    league_rank = 1
    used_data_trunc = 200
    N = 1

    idx = Y_data != league_rank
    X_data_trunc = X_data.loc[idx, :]
    Y_data_trunc = Y_data[idx]

    idx_us = np.where(Y_data == league_rank)[0][:used_data_trunc]
    idx_left = np.where(Y_data == league_rank)[0][used_data_trunc:]


    X_data_us = X_data.loc[idx_us, :]
    Y_data_us = Y_data[idx_us]

    X_data_left = X_data.loc[idx_left, :]
    Y_data_left = Y_data[idx_left]

    X_us = pd.concat([X_data_trunc, X_data_us])
    Y_us = pd.concat([Y_data_trunc, Y_data_us])

    oversample = SMOTE()
    X_os, Y_os = oversample.fit_resample(X_us, Y_us)

    X_new_temp = pd.concat([X_os, X_data_left])
    Y_new_temp = pd.concat([Y_os, Y_data_left])

    X_smote, Y_smote = oversample.fit_resample(X_new_temp, Y_new_temp)

    for _ in range(N):

        idx = Y_smote != league_rank
        X_data_trunc = X_smote.loc[idx, :]
        Y_data_trunc = Y_smote[idx]

        idx_us = np.where(Y_smote == league_rank)[0][:used_data_trunc]
        idx_left = np.where(Y_smote == league_rank)[0][used_data_trunc:]


        X_data_us = X_smote.loc[idx_us, :]
        Y_data_us = Y_smote[idx_us]

        X_data_left = X_smote.loc[idx_left, :]
        Y_data_left = Y_smote[idx_left]

        X_us = pd.concat([X_data_trunc, X_data_us])
        Y_us = pd.concat([Y_data_trunc, Y_data_us])

        X_os, Y_os = oversample.fit_resample(X_us, Y_us)

        X_new_temp = pd.concat([X_os, X_data_left])
        Y_new_temp = pd.concat([Y_os, Y_data_left])

        X_smote, Y_smote = oversample.fit_resample(X_new_temp, Y_new_temp)


    return X_smote, Y_smote

def sort_clubs(X_data):
    #Sort clubs that have the same position
    clubs_id = X_data["club_id"]
    clubs_id = np.array(clubs_id)
    help_data = X_data.drop(columns=["club_id", "league_rank"])

    columns = ["avg_stars_top_11_players", "session_count_last_28_days", "playtime_last_28_days",
               "days_active_last_28_days", "league_match_watched_count_last_28_days"]

    for i in range(len(clubs_id)):
        idx_i = X_data["club_id"] == clubs_id[i]
        max_arr = np.array(help_data.loc[idx_i, columns])
        max_idx = i
        for j in range(i+1, len(clubs_id)):
            idx_j = X_data["club_id"] == clubs_id[j]
            j_arr = np.array(help_data.loc[idx_j, columns])
            if(sum((max_arr > j_arr)[0]) < sum((j_arr > max_arr)[0])):
                max_idx = j
                max_arr = j_arr
        clubs_id[i], clubs_id[max_idx] = clubs_id[max_idx], clubs_id[i]

    return clubs_id

def fix_position(X_data):

    league_ranks = np.unique(X_data["league_rank"])
    possible_ranks = [i for i in range(1, 15)]
    dict_of_ranks = {}

    for league_rank in league_ranks:
        idx = X_data["league_rank"] == league_rank
        if(sum(idx) == 1):
            key = int(X_data.loc[idx, "club_id"])
            dict_of_ranks[key] = possible_ranks.pop(0)
        else:

            clubs_id_sort = sort_clubs(X_data.loc[idx, :])
            for club_sort in clubs_id_sort:
                dict_of_ranks[club_sort] = possible_ranks.pop(0)


    for key, value in dict_of_ranks.items():
        idx = X_data["club_id"] == key
        X_data.loc[idx, "league_rank"] = value

    return X_data

def same_positions(X_data):

    leagues = np.unique(X_data["league_id"])
    for league in leagues:
        idx = X_data["league_id"] == league
        temp_data = X_data.loc[idx, :]
        X_data.loc[idx, :] = fix_position(temp_data)



train_data = pd.read_csv("jobfair_train.csv")
test_data = pd.read_csv("jobfair_test.csv")

X_train, Y_train = prepare_train_data(train_data)

leagues_in_test = test_data["league_id"]
clubs_in_test = test_data["club_id"]
X_test = prepare_test_data(test_data)



X_train_smote, Y_train_smote = make_synthetic_data(X_train, Y_train)



#X_train_new, X_val, Y_train_new, Y_val = train_test_split(X_train_smote, Y_train_smote, test_size=0.3)


#model = LogisticRegression(multi_class="multinomial")
#model.fit(X_train_new, Y_train_new)
#Y_val_pred = model.predict(X_val)


#clf = DecisionTreeClassifier(criterion="entropy")
#clf.fit(X_train_new, Y_train_new)
#Y_val_pred = clf.predict(X_val)

#rf = RandomForestClassifier(criterion="entropy", n_estimators=500)
#rf.fit(X_train_new, Y_train_new)
#Y_val_pred = rf.predict(X_val)

#print("Acc:", metrics.accuracy_score(Y_val, Y_val_pred))
#print("MAE:", metrics.mean_absolute_error(Y_val, Y_val_pred))
#matrix = confusion_matrix(Y_val, Y_val_pred)
#print(matrix)
#sns.heatmap(matrix, annot=True)
#plt.show()


"""param_grid = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 50, 100],
    'max_features': [5, 7, 10],
    'n_estimators': [100, 200, 300, 1000]
}
rf = RandomForestClassifier(criterion="entropy")
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train_new, Y_train_new)
print(grid_search.best_estimator_)"""

rf = RandomForestClassifier(criterion="entropy", n_estimators=500)
rf.fit(X_train_smote, Y_train_smote)
Y_test = rf.predict(X_test)
X_test["club_id"] = clubs_in_test
X_test["league_id"] = leagues_in_test
X_test["league_rank"] = Y_test
same_positions(X_test)
results = X_test.loc[:, ["club_id", "league_rank"]]
results.to_csv("league_rank_predictions.csv")




