import cfbd
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

import get_curr_year

year_list = [2020, 2021, 2022, 2023, 2024]

if __name__ == "__main__":
    configuration = get_curr_year.config()
    week_num = 14 #assuming this gets everything

    with cfbd.ApiClient(configuration) as api_client:
        api_client.default_headers["Authorization"] = f"Bearer {configuration.api_key['authorization']}"
        api_instance_games = cfbd.GamesApi(api_client)
        api_instance_rankings = cfbd.RankingsApi(api_client)
        api_instance_elo = cfbd.RatingsApi(api_client)

        team_game = pd.DataFrame()
        ap_poll_df = pd.DataFrame()
        elo_df = pd.DataFrame()

        for year in year_list:
            for i in range(1, week_num):
                hold = get_curr_year.team_stats_by_game(api_instance_games, year, i)
                team_game = pd.concat([team_game, hold], ignore_index=True)

                hold3 = get_curr_year.get_elo(api_instance_elo, i, year)
                elo_df = pd.concat([elo_df, hold3], ignore_index=True)
            
            team_game["Year"] = year
            elo_df["Year"] = year

        team_game = get_curr_year.label_encoder(team_game, "homeAway")
        team_game = team_game.fillna(0)

        col_list = ["completionAttempts","fourthDownEff","thirdDownEff","totalPenaltiesYards"]
        team_game = get_curr_year.turn_dash_precentage(team_game, col_list)

        print(team_game.head())
        print(elo_df.tail())

        team_game.to_csv("../CFB_predictions_take_2/pre_calc_data/past_years_weekly_game_data.csv", index=False, encoding="utf-8")
        elo_df.to_csv("../CFB_predictions_take_2/pre_calc_data/past_years_weekly_elo_data.csv", index=False, encoding="utf-8")