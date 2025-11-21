import pandas as pd
import joblib


if __name__ == "__main__":
    df = pd.read_csv("../CFB_predictions_take_2/post_calc_data/combined_data.csv")
    model = joblib.load("../CFB_predictions_take_2/saved_models/ensembleModel.pkl")
    scaler = joblib.load("../CFB_predictions_take_2/saved_models/scaler.pkl")

    baylor = df[
        (df["team"] == "Baylor") &
        (df["week"] == 10) &
        (df["Year"] == 2025)
    ]

    utah = df[
        (df["team"] == "Utah") &
        (df["week"] == 10) &
        (df["Year"] == 2025)
    ]

    drop_cols = ["week", "Year", "team", "Win?"]
    baylor = baylor.drop(columns=drop_cols)
    utah = utah.drop(columns=drop_cols)

    utah_team_cols = [c for c in utah.columns if not c.endswith("_opp")]
    utah = utah[utah_team_cols]

    utah_opp = utah.add_suffix("_opp")

    game_row = pd.concat([baylor.reset_index(drop=True), utah_opp.reset_index(drop=True)], axis=1)

    col_order = pd.read_csv("../CFB_predictions_take_2/post_calc_data/model_feature_columns.csv", header=None)[0].tolist()
    game_row = game_row.reindex(columns=col_order, fill_value=0)

    game_row_scaled = scaler.transform(game_row)
    pred = model.predict(game_row_scaled)
    prob = model.predict_proba(game_row_scaled)


