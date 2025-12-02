from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    drop_cols = ["week", "Year", "team"]
    
    df_main = pd.read_csv("../CFB_predictions_take_2/post_calc_data/combined_data.csv")
    df_team = pd.read_csv("../CFB_predictions_take_2/post_calc_data/combined_data.csv")
    df_main = df_main.drop(columns = drop_cols, axis = 0)

    
    y = df_main["Win?"]
    X = df_main.drop(columns = "Win?", axis = 0)

    scaler = MaxAbsScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23, stratify=y
    )

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    preds_log = log_reg.predict(X_test)
    acc_log = accuracy_score(y_test, preds_log)

    
    bama = df_team[
        (df_team["team"] == "Alabama") &
        (df_team["week"] == 11) &
        (df_team["Year"] == 2025)
    ]

    lsu = df_team[
        (df_team["team"] == "LSU") &
        (df_team["week"] == 11) &
        (df_team["Year"] == 2025)
    ]

    lsu = lsu.drop(columns=drop_cols,  axis = 0)
    bama = bama.drop(columns=drop_cols,  axis = 0)
    lsu = lsu.drop(columns="Win?",  axis = 0)
    bama = bama.drop(columns="Win?",  axis = 0)

    bama_team_cols = [c for c in bama.columns if not c.endswith("_opp")]
    bama = bama[bama_team_cols]

    lsu_team_cols = [c for c in lsu.columns if not c.endswith("_opp")]
    lsu = lsu[lsu_team_cols]

    bama_opp = bama.add_suffix("_opp")

    game_row = pd.concat([lsu.reset_index(drop=True), bama_opp.reset_index(drop=True)], axis=1)
    game_row = game_row.drop(columns = "homeAway_opp", axis = 0)

    col_order = pd.read_csv(
        "../CFB_predictions_take_2/post_calc_data/model_feature_columns.csv",
        header=None
    )[0].astype(str).tolist()

    col_order = [c for c in col_order if not c.isdigit()]

    game_row = game_row.reindex(columns=col_order, fill_value=0)
    game_row = scaler.fit_transform(game_row)

    pred = log_reg.predict(game_row)
    prob = log_reg.predict_proba(game_row)
    print(pred)
    print(prob)