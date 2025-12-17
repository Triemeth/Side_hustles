import pandas as pd

def expand_team_opp_rows(df):
    id_cols = ["team", "opp", "week", "Year", "homeAway"]

    team_stats = [c for c in df.columns if not c.endswith("_opp") and c not in id_cols]
    opp_stats = [c for c in df.columns if c.endswith("_opp")]

    df_team = df.copy()
    df_opp = df.copy()

    df_opp["team"], df_opp["opp"] = df["opp"], df["team"]

    for c in team_stats:
        df_opp[c] = df[c + "_opp"]

    for c in opp_stats:
        base = c.replace("_opp", "")
        df_opp[c] = df[base]

    df_out = pd.concat([df_team, df_opp], ignore_index=True)

    return df_out

if __name__ == "__main__":
    df = pd.read_csv("../CFB_predictions_take_2/post_calc_data/combined_data.csv")
    df = expand_team_opp_rows(df)
    print(df.columns)
    df.to_csv("../CFB_predictions_take_2/check_data/check_dup_cols.csv", index=False, encoding="utf-8")
