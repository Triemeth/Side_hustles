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


CURR_YEAR = datetime.now().year

def config():
    load_dotenv('/app/.env')
    api_key = os.getenv("CFD_API_KEY")
    if not api_key:
        raise ValueError("No API key found. Make sure CFD_API_KEY is in your .env file")

    configuration = cfbd.Configuration(
        host="https://api.collegefootballdata.com",
        access_token = api_key
    )

    return configuration

if __name__ == "__main__":
    configuration = config()

    with cfbd.ApiClient(configuration) as api_client:
        api_instance = cfbd.AdjustedMetricsApi(api_client)

        team = 'Baylor'
        conference = "Big 12"

        api_response = api_instance.get_adjusted_team_season_stats(year = CURR_YEAR, team = team, conference = conference)
        pprint(api_response)