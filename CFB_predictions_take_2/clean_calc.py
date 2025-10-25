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

def possesion_time_clean(df):
    df["possessionTimeSeconds"] = pd.to_numeric(df["possessionTime"].str[:2]) * 60 + pd.to_numeric(df["possessionTime"].str[3:])
    df = df.drop("possessionTime", axis = 1)
    return df

#need if 0 dont add ap_strngth
#need to fix all columns as well
def comp_def_score(df):
    if df["ap_poll"] != 0:
        df["defensive_score"] = (
        -0.25 * (df["totalYardsOpponent"] / df["games"]) -
        0.20 * (7 * (df["passingTDsOpponent"] + df["rushingTDsOpponent"])) +
        0.15 * (df["turnoversOpponent"] / df["games"]) +
        0.10 * (df["sacks"] / df["games"]) -
        0.10 * (df["thirdDownConversionsOpponent"] / df["thirdDownsOpponent"]).replace([np.inf, np.nan], 0) -
        0.10 * (df["fourthDownConversionsOpponent"] / df["fourthDownsOpponent"]).replace([np.inf, np.nan], 0) -
        0.10 * (df["penaltyYardsOpponent"] / df["games"]) +
        0.10 * df["ap_strength"])
    else:
        df["defensive_score"] = (
        -0.25 * (df["totalYardsOpponent"] / df["games"]) -
        0.20 * (7 * (df["passingTDsOpponent"] + df["rushingTDsOpponent"])) +
        0.15 * (df["turnoversOpponent"] / df["games"]) +
        0.10 * (df["sacks"] / df["games"]) -
        0.10 * (df["thirdDownConversionsOpponent"] / df["thirdDownsOpponent"]).replace([np.inf, np.nan], 0) -
        0.10 * (df["fourthDownConversionsOpponent"] / df["fourthDownsOpponent"]).replace([np.inf, np.nan], 0) -
        0.10 * (df["penaltyYardsOpponent"] / df["games"]))

    return df

#need if 0 dont add ap_strngth
#need to fix all columns as well
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
    games_dat = pd.read_csv("../CFB_predictions_take_2/pre_calc_data/weekly_game_data.csv")
    ap_poll_dat = pd.read_csv("../CFB_predictions_take_2/pre_calc_data/weekly_ap_poll_data.csv")

    games_dat = clean_game_dat(games_dat)

    ap_poll_dat["ap_strength"] = ap_strength_bonus(ap_poll_dat["ap_rank"])
    ap_poll_dat = clean_ap_poll(ap_poll_dat)

    combined_data = pd.merge(games_dat, ap_poll_dat, left_on = ["team", "week"], right_on = ["team", "week"], how = "left")
    combined_data = possesion_time_clean(combined_data)

    rolling_cols = ['completionAttempts', 'defensiveTDs', 'firstDowns', 'fourthDownEff', 
                'fumblesLost', 'fumblesRecovered', 'interceptionTDs', 
                'interceptionYards', 'interceptions','kickReturnTDs', 'kickReturnYards', 
                'kickReturns', 'kickingPoints','netPassingYards', 'passesDeflected',
                'passesIntercepted', 'passingTDs','points', 'possessionTimeSeconds', 
                'puntReturnTDs', 'puntReturnYards','puntReturns', 'qbHurries',
                'rushingAttempts', 'rushingTDs', 'rushingYards', 'sacks',
                'tackles', 'tacklesForLoss', 'thirdDownEff', 'totalFumbles', 
                'totalPenaltiesYards', 'totalYards','turnovers', 'yardsPerPass', 
                'yardsPerRushAttempt']
    
    for col in rolling_cols:
        combined_data[f"{col}_rolling_avg"] = combined_data.groupby("team")[col].rolling(window=3).mean().reset_index(level=0, drop=True)
        combined_data[f"{col}_rolling_sum"] = combined_data.groupby("team")[col].rolling(window=3).sum().reset_index(level=0, drop=True)

    combined_data["ap_strength"] = combined_data["ap_strength"].fillna(0)
    combined_data["ap_rank"] = combined_data["ap_rank"].fillna(0)

    combined_data = combined_data.dropna()

    combined_data = combined_data.drop(columns = rolling_cols, axis = 1)

    combined_data.to_csv("../CFB_predictions_take_2/pre_calc_data/combined_data.csv", index=False, encoding="utf-8")

    #combined_data = comp_def_score(combined_data)
    #combined_data = comp_off_score(combined_data)