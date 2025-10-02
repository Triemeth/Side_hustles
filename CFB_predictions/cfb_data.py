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

def season_team_stats( api_client):
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


if __name__ == "__main__":
    configuration = config()

    with cfbd.ApiClient(configuration) as api_client:
        api_client.default_headers["Authorization"] = f"Bearer {configuration.api_key['authorization']}"

        year = CURR_YEAR
        team = "Baylor"

        #need to do rolling averages as well and homeverse away formula

        
        
        # will need to later loop through all fbs teams and append to a main dataframe
        #df_team = season_team_stats(api_client)


