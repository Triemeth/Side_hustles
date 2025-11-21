import pandas as pd
import numpy as np
from datetime import datetime

CURR_YEAR = datetime.now().year

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

def bin_win_loss(df):
    df["Win?"] = np.where(df["points"] > df["points_opp"], 1, 0)
    return df

def off_score(df):
    
    offense_metrics = ["totalYards", "netPassingYards", "passingTDs",
                       "rushingTDs", "yardsPerPass", "yardsPerRushAttempt",
                       "thirdDownEff", "possessionTimeSeconds", "points",
                       "kickingPoints", "rushingAttempts", "sacks",
                       "turnovers", "totalPenaltiesYards", "elo_opp"]
    
    #used chatgpt to generate these will probably need to mess with a lil
    offense_weights = [
        0.10,   # totalYards
        0.05,   # netPassingYards
        0.15,   # passingTDs
        0.10,   # rushingTDs
        0.10,   # yardsPerPass
        0.10,   # yardsPerRushAttempt
        0.10,   # thirdDownEff
        0.05,   # possessionTimeSeconds
        0.15,   # points
        0.02,   # kickingPoints
        0.03,   # rushingAttempts
        -0.05,  # sacks (bad)
        -0.12,  # turnovers (very bad)
        -0.08,  # totalPenaltiesYards (bad)
        0.05    # ELO (this is so low due to elo being in thousands honesly still probably too large likely need .0001)
    ]


    off_score = (
        offense_weights[0]  * df["totalYards"] +
        offense_weights[1]  * df["netPassingYards"] +
        offense_weights[2]  * df["passingTDs"] +
        offense_weights[3]  * df["rushingTDs"] +
        offense_weights[4]  * df["yardsPerPass"] +
        offense_weights[5]  * df["yardsPerRushAttempt"] +
        offense_weights[6]  * df["thirdDownEff"] +
        offense_weights[7]  * df["possessionTimeSeconds"] +
        offense_weights[8]  * df["points"] +
        offense_weights[9]  * df["kickingPoints"] +
        offense_weights[10]  * df["rushingAttempts"] +
        offense_weights[11]  * df["sacks"] +
        offense_weights[12]  * df["turnovers"] +
        offense_weights[13]  * df["totalPenaltiesYards"] +
        offense_weights[14]  * df["elo_opp"]
    )
    
    df["off_score"] = off_score

    return df["off_score"]

def def_score(df):

    defense_metrics = ["defensiveTDs", "interceptions", "interceptionYards",
                       "fumblesRecovered", "sacks", "tackles", "tacklesForLoss",
                       "passesDeflected", "qbHurries", "kickReturnTDs", "elo_opp"]
    
    #used chatgpt to generate these will probably need to mess with a lil
    defense_weights = [
        0.15,   # defensiveTDs (rare & high impact)
        0.20,   # interceptions
        0.05,   # interceptionYards (value, but minor)
        0.15,   # fumblesRecovered
        0.10,   # sacks (pressure)
        0.05,   # tackles (base volume stabilizer)
        0.10,   # tacklesForLoss (disruptive plays)
        0.08,   # passesDeflected (pass disruption)
        0.07,   # qbHurries (pressure indicator)
        0.03,   # kickReturnTDs (rare boost)
        0.05    # ELO (this is so low due to elo being in thousands honesly still probably too large likely need .0001)
    ]
    
    def_score = (
        defense_weights[0]  * df["defensiveTDs"] +
        defense_weights[1]  * df["interceptions"] +
        defense_weights[2]  * df["interceptionYards"] +
        defense_weights[3]  * df["fumblesRecovered"] +
        defense_weights[4]  * df["sacks"] +
        defense_weights[5]  * df["tackles"] +
        defense_weights[6]  * df["tacklesForLoss"] +
        defense_weights[7]  * df["passesDeflected"] +
        defense_weights[8]  * df["qbHurries"] +
        defense_weights[9]  * df["kickReturnTDs"] +
        defense_weights[10]  * df["elo_opp"]
    )

    df["def_score"] = def_score

    return df["def_score"]

def rolling_aggs(df, cols):
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    for col in cols:
        df[f"{col}_rolling_avg"] = (
            df.groupby("team")[col]
              .shift(1) 
              .rolling(window=3, min_periods=1)
              .mean()
        )

    #shift should not grab the current week without it scores better but think it was cheating 
    #for col in cols: rolled = ( df.groupby("team")[col] .rolling(window=3, min_periods=1) .mean() .reset_index(level=0, drop=True) ) df[f"{col}_rolling_avg"] = rolled

    return df

if __name__ == "__main__":
    curr_games_dat = pd.read_csv("../CFB_predictions_take_2/pre_calc_data/weekly_game_data.csv")
    curr_elo_dat = pd.read_csv("../CFB_predictions_take_2/pre_calc_data/weekly_elo_data.csv")
    past_games_dat = pd.read_csv("../CFB_predictions_take_2/pre_calc_data/past_years_weekly_game_data.csv")
    past_elo_dat = pd.read_csv("../CFB_predictions_take_2/pre_calc_data/past_years_weekly_elo_data.csv")

    curr_games_dat["Year"] = CURR_YEAR

    curr_games_dat = clean_game_dat(curr_games_dat)
    past_games_dat = clean_game_dat(past_games_dat)

    combined_data_curr = pd.merge(curr_games_dat, curr_elo_dat, left_on = ["team", "week"], right_on = ["team", "week"], how = "left")

    combined_data_past = pd.merge(past_games_dat, past_elo_dat, left_on = ["team", "week", "Year"], right_on = ["team", "week", "Year"], how = "left")

    combined_data = pd.concat([combined_data_past, combined_data_curr], ignore_index=True)
    combined_data = possesion_time_clean(combined_data)

    #Where off and def scores used to be

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
    
    combined_data = rolling_aggs(combined_data, rolling_cols)

    combined_data = combined_data.merge(combined_data, left_on = ["gameId", "team", "Year"], right_on = ["gameId", "team_opp", "Year"], suffixes = ("", "_opp"))

    combined_data = combined_data.sort_values(["team", "week"]).copy()
    combined_data["points_opp"] = pd.to_numeric(combined_data["points_opp"], errors="coerce").fillna(0)

    combined_data["points_scored_against_rolling_avg"] = (
        combined_data.groupby("team")["points_opp"]
        .shift(1).transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    #I am still not sure if this is the best spot for this
    #needed to be post self merge for elo_opp so going to have to do another rolling aggregation for off and def score going to throw elo in here too
    combined_data["off_score"] = off_score(combined_data)
    combined_data["def_score"] = def_score(combined_data)

    rolling_cols2 = ["off_score", "def_score", "elo"]
    combined_data = rolling_aggs(combined_data, rolling_cols2)

    combined_data = combined_data.dropna()
    point = combined_data["points"]
    combined_data = combined_data.drop(columns = rolling_cols, axis = 1)
    combined_data["points"] = point

    combined_data = bin_win_loss(combined_data)

    more_cols_to_drop = ["date", "gameId", "week", "Year", "conference",
                    "conference_opp", "elo", "elo_opp", "possessionTimeSeconds_opp", 
                    "points", "points_opp", "off_score", "def_score", "date_opp", 
                    "homeAway_opp", "team_opp", "week_opp", "team_opp_opp"]
    combined_data = combined_data.drop(columns = more_cols_to_drop, axis = 0)

    combined_data.to_csv("../CFB_predictions_take_2/post_calc_data/combined_data.csv", index=False, encoding="utf-8")

