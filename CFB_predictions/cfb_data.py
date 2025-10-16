import cfbd
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

CURR_YEAR = datetime.now().year

def config():
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("CFBD_API_KEY")

    configuration = cfbd.Configuration(
        host="https://api.collegefootballdata.com"
    )

    configuration.api_key = {}
    configuration.api_key_prefix = {}

    configuration.api_key["authorization"] = api_key
    configuration.api_key_prefix["authorization"] = "Bearer"

    return configuration

def label_encoder(df, col_name):
    le = LabelEncoder()
    df[col_name] = le.fit_transform(df[col_name])

    return df

def season_team_stats(api_instance, year = CURR_YEAR):
    game_stats = api_instance.get_team_stats(year = year)

    data = []
    for stat in game_stats:
        data.append({
            "team": stat.team,
            "stat_name": stat.stat_name,
            "value": stat.stat_value.actual_instance
        })

    df = pd.DataFrame(data)
    df = df.pivot(index="team", columns="stat_name", values="value").reset_index()

    return df

def turn_dash_precentage(df, col_list):
    for col in col_list:
        df[col] = df[col].astype("string")

        split_vals = df[col].str.extract(r"(\d+)-(\d+)")
        num = pd.to_numeric(split_vals[0], errors="coerce")
        den = pd.to_numeric(split_vals[1], errors="coerce")
        
        df[col] = np.where((den.notna()) & (den != 0), num / den, 0)

    return df

def team_stats_by_game(api_instance, year=CURR_YEAR, week = 1):

    games = api_instance.get_game_team_stats(year = year, week = week)
    games_info = api_instance_games.get_games(year=year, week=week)
    game_dates = {g.id: g.start_date for g in games_info}

    parsed_games = []

    for g in games:
        g_dict = g.to_dict()
        game_id = g_dict.get("id")
        game_date = game_dates.get(game_id)

        for t in g_dict.get("teams", []):
            base = {
                "gameId": game_id,
                "date": game_date,
                "team": t.get("team"),
                "teamId": t.get("team_id"),
                "conference": t.get("conference"),
                "homeAway": t.get("homeAway"),
                "points": t.get("points"),
            }

            stats_dict = {}
            for s in t.get("stats", []):
                cat = s.get("category")
                val = s.get("stat")
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    pass
                stats_dict[cat] = val

            parsed_games.append({**base, **stats_dict})

    df = pd.DataFrame(parsed_games)
    df = df.reindex(sorted(df.columns), axis=1)

    return df

def get_ap_poll(api_instance, year = CURR_YEAR):
    rankings = api_instance.get_rankings(year=year, week = 1)

    latest = max(rankings, key=lambda x: x.week)
    ap_poll = next((poll for poll in latest.polls if poll.poll == "AP Top 25"), None)

    if not ap_poll:
        return pd.DataFrame(columns=["team", "ap_rank"])

    data = []
    for team in ap_poll.ranks:
        data.append({
            "team": team.school,
            "conference": team.conference,
            "ap_rank": team.rank
        })

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    configuration = config()

    with cfbd.ApiClient(configuration) as api_client:
        api_client.default_headers["Authorization"] = f"Bearer {configuration.api_key['authorization']}"
        api_instance_games = cfbd.GamesApi(api_client)
        api_instance_season = cfbd.StatsApi(api_client)
        api_instance_rankings = cfbd.RankingsApi(api_client)

        team_game = pd.DataFrame()

        for i in range(1,7):
            hold = team_stats_by_game(api_instance_games, CURR_YEAR, i)
            team_game = pd.concat([team_game, hold], ignore_index=True)

        team_game = label_encoder(team_game, "homeAway")
        team_game = team_game.fillna(0)

        col_list = ["completionAttempts","fourthDownEff","thirdDownEff","totalPenaltiesYards"]
        team_game = turn_dash_precentage(team_game, col_list)

        team_season = season_team_stats(api_instance_season)
        team_season = team_season.fillna(0)

        ap_poll = get_ap_poll(api_instance_rankings)

        team_game.to_csv("../CFB_predictions/pre_calc_data/team_game_dat.csv", index=False, encoding="utf-8")
        team_season.to_csv("../CFB_predictions/pre_calc_data/team_season_dat.csv", index=False, encoding="utf-8")
        ap_poll.to_csv("../CFB_predictions/pre_calc_data/ap_poll_dat.csv", index=False, encoding="utf-8")