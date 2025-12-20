import pandas as pd

def dup_cols(df):
    opp_cols = [c for c in df.columns if c.endswith('_opp')]
    norm_cols = [c for c in df.columns if not c.endswith('_opp')]

    opp_cols += ['week', 'Year']

    df_opp_w_suffix = df[opp_cols]
    df_norm_no_suffix = df[norm_cols]

    df_opp_no_suffix = df_opp_w_suffix.copy()
    df_opp_no_suffix.columns = df_opp_w_suffix.columns.str.removesuffix('_opp')
    df_norm_w_suffix = df_norm_no_suffix.add_suffix('_opp', axis = 1)

    df_main = pd.merge(df_norm_no_suffix, df_opp_no_suffix, on = ["team", 'week', 'Year'], how = 'inner')
    return df_main

if __name__ == "__main__":
    df = pd.read_csv("../CFB_predictions_take_2/post_calc_data/combined_data.csv")
    df = dup_cols(df)
    print(df.columns)
    df.to_csv("../CFB_predictions_take_2/check_data/check_dup_cols.csv", index=False, encoding="utf-8")
