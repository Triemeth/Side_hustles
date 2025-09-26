import time
import cfbd
from cfbd.rest import ApiException
from pprint import pprint
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
from cfbd.models.adjusted_team_metrics import AdjustedTeamMetrics
from pprint import pprint
from pathlib import Path

CURR_YEAR = datetime.now().year

def config():
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("CFBD_API_KEY")

    configuration = cfbd.Configuration(
        host="https://api.collegefootballdata.com"
    )

    configuration.api_key["Authorization"] = api_key.strip()
    configuration.api_key_prefix["Authorization"] = "Bearer"

    return configuration

if __name__ == "__main__":
    configuration = config()

    with cfbd.ApiClient(configuration) as api_client:
        games_api = cfbd.GamesApi(api_client)
        games = games_api.get_games(year=2024, team="Baylor")
        print(games[:1])
