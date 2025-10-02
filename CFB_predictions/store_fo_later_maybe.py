# didnt work yet and may be less effiecent then another way


# api_instance = cfbd.GamesApi(api_client)

#         for week in range(1, 6):
#             game_stats = api_instance.get_game_team_stats(year = year, week = week)

#             data = [
#                 {
#                     "game_id": game.id,
#                     "game_date": game.start_date,
#                     "team": team_stat.school,
#                     "opponent": team_stat.opponent,
#                     "home_away": "Home" if team_stat.home_away == "home" else "Away",
#                     "stat_name": stat.stat_name,
#                     "value": stat.stat_value.actual_instance
#                 }
#                 for game in game_stats
#                 for team_stat in game.teams
#                 for stat in team_stat.stats
#             ]
                
#             df = pd.DataFrame(data)

#             df = df.pivot_table(
#                 index=["game_id", "game_date", "team", "opponent", "home_away"],
#                 columns="stat_name",
#                 values="value",
#                 aggfunc="first"
#             ).reset_index()
            
#         print(df.head(5))