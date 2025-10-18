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

def clean_ap_poll(df):
    df = df.drop(columns = ["conference"], axis = 1)
    return df

if __name__ == "__main__":
    games_dat = pd.read_csv("../CFB_predictions_take_2/pre_calc_data/weekly_game_data.csv")
    ap_poll_dat = pd.read_csv("../CFB_predictions_take_2/pre_calc_data/weekly_ap_poll_data.csv")

    games_dat = clean_game_dat(games_dat)

    ap_poll_dat["ap_strength"] = ap_strength_bonus(ap_poll_dat["ap_rank"])
    ap_poll_dat = clean_ap_poll(ap_poll_dat)

    combined_data = pd.merge(games_dat, ap_poll_dat, left_on = ["team", "week"], right_on = ["team", "week"], how = "left")
    dat_check = combined_data[combined_data["ap_rank"].notna()]
    print(dat_check.head())
    print(combined_data.head())