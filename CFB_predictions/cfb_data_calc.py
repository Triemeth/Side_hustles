import pandas as pd
import numpy as np

def ap_strength_bonus(rank):
    return (26 - rank) / 25 

def clean_game_dat(df):
    df.columns = df.columns.str.strip()

    df = df.merge(df[["gameId", "team"]], on = "gameId", suffixes=('', "_opp"))
    df = df[df["team"] != df["team_opp"]]

    keep_conf = ["American Athletic", "ACC", 
                     "Big 12", "Big Ten", 
                     "Conference USA", "FBS Independents", 
                     "Mid-American", "Mountain West",
                     "Pac-12", "SEC", "Sun Belt"]
    df = df[df["conference"].isin(keep_conf)]

    cols_to_delete = ["conference", "teamId"]
    df = df.drop(columns=cols_to_delete, axis = 1)

    return df

def comp_def_score(df):
    df["defensive_score"] = (
    -0.25 * (df["totalYardsOpponent"] / df["games"]) -
    0.20 * (7 * (df["passingTDsOpponent"] + df["rushingTDsOpponent"])) +
    0.15 * (df["turnoversOpponent"] / df["games"]) +
    0.10 * (df["sacks"] / df["games"]) -
    0.10 * (df["thirdDownConversionsOpponent"] / df["thirdDownsOpponent"]).replace([np.inf, np.nan], 0) -
    0.10 * (df["fourthDownConversionsOpponent"] / df["fourthDownsOpponent"]).replace([np.inf, np.nan], 0) -
    0.10 * (df["penaltyYardsOpponent"] / df["games"]) +
    0.10 * df["ap_strength"])

    return df

def comp_off_score(df):
    df["offensive_score_raw"] = (
        0.25 * (df["totalYards"] / df["games"]) +
        0.20 * (7 * (df["passingTDs"] + df["rushingTDs"])) +
        0.15 * ((df["thirdDownConversions"] / df["thirdDowns"]).replace([np.inf, np.nan], 0)) +
        0.10 * ((df["fourthDownConversions"] / df["fourthDowns"]).replace([np.inf, np.nan], 0)) +
        0.10 * (df["possessionTime"] / df["games"]) -
        0.10 * (df["turnovers"] / df["games"]) -
        0.10 * (df["penaltyYards"] / df["games"])
    )

    df["offensive_score_adj"] = df["offensive_score_raw"] * (1 + 0.20 * df["ap_strength"])

    return df


if __name__ == "__main__":
    games_dat = pd.read_csv("../CFB_predictions/pre_calc_data/team_game_dat.csv")
    season_dat = pd.read_csv("../CFB_predictions/pre_calc_data/team_season_dat.csv")
    ap_poll_dat = pd.read_csv("../CFB_predictions/pre_calc_data/ap_poll_dat.csv")

    games_dat = clean_game_dat(games_dat)

    ap_poll_dat["ap_strength"] = ap_strength_bonus(ap_poll_dat["ap_rank"])

    season_dat = pd.merge(season_dat, ap_poll_dat[['team', 'ap_strength']], on = "team", how = "left")
    season_dat = season_dat.fillna(0)

    season_dat = comp_def_score(season_dat)
    season_dat = comp_off_score(season_dat)

    off_sort_season = season_dat.sort_values(by = "offensive_score_adj", ascending=False)
    def_sort_season = season_dat.sort_values(by = "defensive_score", ascending=False)

    print("off scores:")
    print(off_sort_season["team"].head())
    print("\n def scores:")
    print(def_sort_season["team"].head())
