import cfbd
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
from pathlib import Path
from pprint import pprint

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


#get team seasons stats up to upcoming games
def season_team_stats(api_client, team, year = CURR_YEAR):
    api_instance = cfbd.StatsApi(api_client)

    game_stats = api_instance.get_team_stats(year = year, team = team)

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

#get game stats mainly for rolling averages and home and away stuff



#data output is monthly for some values due to excel transforming stuff like 11-40 to nov-40 or something need to regex to fix this or turn to precentage
#could do SQL lowkey but not worth having some of the odd values as strings tbh
if __name__ == "__main__":
    configuration = config()

    with cfbd.ApiClient(configuration) as api_client:
        api_client.default_headers["Authorization"] = f"Bearer {configuration.api_key['authorization']}"
        api_instance = cfbd.GamesApi(api_client)

        games = api_instance.get_game_team_stats(year=CURR_YEAR, week=1)
        parsed_games = []

        for g in games:
            g_dict = g.to_dict()
            for t in g_dict.get("teams", []):
                base = {
                    "gameId": g_dict.get("id"),
                    "team": t.get("team"),
                    "teamId": t.get("team_id"),
                    "conference": t.get("conference"),
                    "homeAway": t.get("home_away"),
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

        print(df.head(10))

        # df.to_csv("../CFB_predictions/check_dat.csv", index=False, encoding="utf-8")
        # print(f"Wrote {len(df)} rows to check_dat.csv")