import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
        0.01    # ELO (this is so low due to elo being in thousands honesly still probably too large likely need .0001)
    ]

    scaler = StandardScaler()
    df_scaled = df.copy()
    
    df_scaled[offense_metrics] = scaler.fit_transform(df[offense_metrics])

    off_score = (
        offense_weights[0]  * df_scaled["totalYards"] +
        offense_weights[1]  * df_scaled["netPassingYards"] +
        offense_weights[2]  * df_scaled["passingTDs"] +
        offense_weights[3]  * df_scaled["rushingTDs"] +
        offense_weights[4]  * df_scaled["yardsPerPass"] +
        offense_weights[5]  * df_scaled["yardsPerRushAttempt"] +
        offense_weights[6]  * df_scaled["thirdDownEff"] +
        offense_weights[7]  * df_scaled["possessionTimeSeconds"] +
        offense_weights[8]  * df_scaled["points"] +
        offense_weights[9]  * df_scaled["kickingPoints"] +
        offense_weights[10]  * df_scaled["rushingAttempts"] +
        offense_weights[11]  * df_scaled["sacks"] +
        offense_weights[12]  * df_scaled["turnovers"] +
        offense_weights[13]  * df_scaled["totalPenaltiesYards"] +
        offense_weights[14]  * df_scaled["elo_opp"]
    )
    
    df_scaled["off_score"] = off_score

    return df_scaled["off_score"]

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
        0.01    # ELO (this is so low due to elo being in thousands honesly still probably too large likely need .0001)
    ]
    
    scaler = StandardScaler()
    df_scaled = df.copy()

    df_scaled[defense_metrics] = scaler.fit_transform(df_scaled[defense_metrics])

    def_score = (
        defense_weights[0]  * df_scaled["defensiveTDs"] +
        defense_weights[1]  * df_scaled["interceptions"] +
        defense_weights[2]  * df_scaled["interceptionYards"] +
        defense_weights[3]  * df_scaled["fumblesRecovered"] +
        defense_weights[4]  * df_scaled["sacks"] +
        defense_weights[5]  * df_scaled["tackles"] +
        defense_weights[6]  * df_scaled["tacklesForLoss"] +
        defense_weights[7]  * df_scaled["passesDeflected"] +
        defense_weights[8]  * df_scaled["qbHurries"] +
        defense_weights[9]  * df_scaled["kickReturnTDs"] +
        defense_weights[10]  * df_scaled["elo_opp"]
    )

    df_scaled["def_score"] = def_score

    return df_scaled["def_score"]

def rolling_aggs(df, rolling_cols):
    
    for col in rolling_cols:
        df[f"{col}_rolling_avg"] = (
            df.groupby("team")[col]
            .rolling(window=3)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return df

if __name__ == "__main__":
    games_dat = pd.read_csv("../CFB_predictions_take_2/pre_calc_data/weekly_game_data.csv")
    ap_poll_dat = pd.read_csv("../CFB_predictions_take_2/pre_calc_data/weekly_ap_poll_data.csv")
    elo_dat = pd.read_csv("../CFB_predictions_take_2/pre_calc_data/weekly_elo_data.csv")

    games_dat = clean_game_dat(games_dat)

    #kinda usless with introduction of elo
    ap_poll_dat["ap_strength"] = ap_strength_bonus(ap_poll_dat["ap_rank"])
    ap_poll_dat = clean_ap_poll(ap_poll_dat)

    combined_data = pd.merge(games_dat, ap_poll_dat, left_on = ["team", "week"], right_on = ["team", "week"], how = "left")
    combined_data = pd.merge(combined_data, elo_dat, left_on = ["team", "week"], right_on = ["team", "week"], how = "left")
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
    """for col in rolling_cols:
        combined_data[f"{col}_rolling_avg"] = (
            combined_data.groupby("team")[col]
            .rolling(window=3)
            .mean()
            .reset_index(level=0, drop=True)
        )"""
        
        

    combined_data["ap_strength"] = combined_data["ap_strength"].fillna(0)
    combined_data["ap_rank"] = combined_data["ap_rank"].fillna(0)

    combined_data = combined_data.merge(combined_data, left_on = ["gameId", "team"], right_on = ["gameId", "team_opp"], suffixes = ("", "_opp"))

    combined_data = combined_data.sort_values(["team", "week"]).copy()
    combined_data["points_opp"] = pd.to_numeric(combined_data["points_opp"], errors="coerce").fillna(0)

    combined_data["points_scored_against_rolling_avg"] = (
        combined_data.groupby("team")["points_opp"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    #I am still not sure if this is the best spot for this
    #needed to be post self merge for elo_opp so going to have to do another rolling aggregation for off and def score going to throw elo in here too
    combined_data["off_score"] = off_score(combined_data)
    combined_data["def_score"] = def_score(combined_data)

    rolling_cols_no_drop = ["off_score", "def_score", "elo"]

    combined_data = rolling_aggs(combined_data, rolling_cols_no_drop)
    """for col in rolling_cols_no_drop:
        combined_data[f"{col}_rolling_avg"] = (
            combined_data.groupby("team")[col]
            .rolling(window=3)
            .mean()
            .reset_index(level=0, drop=True)
        )"""


    combined_data = combined_data.dropna()
    point = combined_data["points"]
    combined_data = combined_data.drop(columns = rolling_cols, axis = 1)
    combined_data["points"] = point

    cols_to_drop = ["date_opp", "homeAway_opp", "team_opp", "week_opp", "team_opp_opp"]
    combined_data = combined_data.drop(columns = cols_to_drop, axis = 1)

    combined_data = bin_win_loss(combined_data)
    #combined_data["off_score"] = comp_off_score(combined_data)

    combined_data.to_csv("../CFB_predictions_take_2/post_calc_data/combined_data.csv", index=False, encoding="utf-8")

    # sum_opp_cols = [col for col in combined_data.columns if "_sum_opp" in col]
    # avg_opp_cols = [col for col in combined_data.columns if "_avg_opp" in col]
    # sum_cols = [col for col in combined_data.columns if "_sum" in col]
    # avg_cols = [col for col in combined_data.columns if "_avg" in col]
