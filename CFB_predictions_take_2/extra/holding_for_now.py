#get curr year
"""def get_ap_poll(api_instance, week, year = CURR_YEAR):
    rankings = api_instance.get_rankings(year=year, week = week)

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
    df["week"] = week

    return df

api_instance_rankings = cfbd.RankingsApi(api_client)

ap_poll_df = pd.DataFrame()

hold2 = get_ap_poll(api_instance_rankings, i, CURR_YEAR)
ap_poll_df = pd.concat([ap_poll_df, hold2], ignore_index=True)

ap_poll_df.to_csv("../CFB_predictions_take_2/pre_calc_data/weekly_ap_poll_data.csv", index=False, encoding="utf-8")"""

#clean calc
"""ap_poll_dat = pd.read_csv("../CFB_predictions_take_2/pre_calc_data/weekly_ap_poll_data.csv")
#kinda usless with introduction of elo
ap_poll_dat["ap_strength"] = ap_strength_bonus(ap_poll_dat["ap_rank"])
ap_poll_dat = clean_ap_poll(ap_poll_dat)
combined_data = pd.merge(games_dat, ap_poll_dat, left_on = ["team", "week"], right_on = ["team", "week"], how = "left")
combined_data["ap_strength"] = combined_data["ap_strength"].fillna(0)
combined_data["ap_rank"] = combined_data["ap_rank"].fillna(0)"""
