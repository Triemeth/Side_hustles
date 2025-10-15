import pandas as pd
import numpy as np

col_list = ["completionAttempts","fourthDownEff","thirdDownEff","totalPenaltiesYards"]

df = pd.read_csv("../CFB_predictions/data/team_game_dat.csv")
df = df[col_list]

for col in col_list:
    df[col] = df[col].astype("string")

for col in col_list:
    split_vals = df[col].str.extract(r"(\d+)-(\d+)")
    num = pd.to_numeric(split_vals[0], errors="coerce")
    den = pd.to_numeric(split_vals[1], errors="coerce")
    
    df[col] = np.where((den.notna()) & (den != 0), num / den, 0)


print(df.head())
print(df.dtypes)