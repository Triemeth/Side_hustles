import pandas as pd
import joblib


if __name__ == "__main__":
    df = pd.read_csv("../CFB_predictions_take_2/post_calc_data/combined_data.csv")
    model = joblib.load("../CFB_predictions_take_2/saved_models/ensembleModel.pkl")
    scaler = joblib.load("../CFB_predictions_take_2/saved_models/scaler.pkl")

    bama = df[
        (df["team"] == "LSU") &
        (df["week"] == 11) &
        (df["Year"] == 2025)
    ]

    lsu = df[
        (df["team"] == "Alabama") &
        (df["week"] == 11) &
        (df["Year"] == 2025)
    ]

    drop_cols = ["week", "Year", "team", "Win?"]
    lsu = lsu.drop(columns=drop_cols)
    bama = bama.drop(columns=drop_cols)

    bama_team_cols = [c for c in bama.columns if not c.endswith("_opp")]
    bama = bama[bama_team_cols]

    lsu_team_cols = [c for c in lsu.columns if not c.endswith("_opp")]
    lsu = lsu[lsu_team_cols]

    bama_opp = bama.add_suffix("_opp")

    game_row = pd.concat([lsu.reset_index(drop=True), bama_opp.reset_index(drop=True)], axis=1)
    game_row.to_csv("../CFB_predictions_take_2/check_data/game_row_check.csv", index=False, encoding="utf-8")
    game_row = game_row.drop(columns = "homeAway_opp", axis = 0)

    col_order = pd.read_csv(
        "../CFB_predictions_take_2/post_calc_data/model_feature_columns.csv",
        header=None
    )[0].astype(str).tolist()

    col_order = [c for c in col_order if not c.isdigit()]

    game_row = game_row.reindex(columns=col_order, fill_value=0)

    game_row_scaled = scaler.transform(game_row)
    pred = model.predict(game_row_scaled)
    prob = model.predict_proba(game_row_scaled)

    print(pred)
    print(prob)


