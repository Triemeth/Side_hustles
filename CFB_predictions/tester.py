import pandas as pd

df = pd.read_csv("../CFB_predictions/check_dat.csv")
df = df[["completionAttempts","fourthDownEff","thirdDownEff","totalPenaltiesYards"]]

print(df.head())
print(len(df))