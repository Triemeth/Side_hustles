import pandas as pd
import joblib


if __name__ == "__main__":
    df = pd.read_csv("../CFB_predictions_take_2/post_calc_data/combined_data.csv")
    model = joblib.load("../CFB_predictions_take_2/saved_models/ensembleModel.pkl")
    scaler = joblib.load("../CFB_predictions_take_2/saved_models/scaler.pkl")

    #df = df.drop(columns = "team_opp.1", axis = 0)

    team = df[
        (df["team"] == "LSU") &
        (df["week"] == 11) &
        (df["Year"] == 2025)
    ]

    opp = df[
        (df["team_opp"] == "Alabama") &
        (df["week"] == 11) &
        (df["Year"] == 2025)
    ]

    drop_cols = ["week", "Year", "team", "Win?", "team_opp"]
    team = team.drop(columns=drop_cols)
    opp = opp.drop(columns=drop_cols)

    opp_cols = [c for c in opp.columns if c.endswith("_opp")]
    opp = opp[opp_cols]

    team_cols = [c for c in team.columns if not c.endswith("_opp")]
    team = team[team_cols]

    game_row = pd.concat([team.reset_index(drop=True), opp.reset_index(drop=True)], axis=1)

    col_order = pd.read_csv(
        "../CFB_predictions_take_2/post_calc_data/model_feature_columns.csv",
        header=None
    )[0].astype(str).tolist()

    col_order = [c for c in col_order if not c.isdigit()]

    game_row = game_row.reindex(columns=col_order, fill_value=0)
    game_row.to_csv("../CFB_predictions_take_2/check_data/game_row_check.csv", index=False, encoding="utf-8")

    #print(game_row[opp_cols].head())
    #print(game_row[team_cols].head())
    #print(opp_cols)
    #print(team_cols)

    game_row_scaled = scaler.transform(game_row)
    pred = model.predict(game_row_scaled)
    prob = model.predict_proba(game_row_scaled)

    print(pred)
    print(prob)


