import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import xgboost as xgb


if __name__ == "__main__":
    df = pd.read_csv("../CFB_predictions_take_2/post_calc_data/combined_data.csv")

    cols_to_drop = ["date", "gameId", "team", "week", "Year", "conference",
                    "conference_opp", "elo", "elo_opp", "possessionTimeSeconds_opp", 
                    "points", "points_opp"]
    df = df.drop(columns = cols_to_drop, axis = 0)

    y = df["Win?"]
    X = df.drop(columns = "Win?", axis = 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .2, random_state = 23)

    model = xgb.XGBClassifier(objective = 'binary:logistic', random_state=23)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(acc)