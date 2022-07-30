import pandas as pd

raw_data = pd.read_csv("raw.tsv", sep = "\t")
out = open("relevant_smiles.txt", "a")

for data_row in range(raw_data.shape[0]):
    if raw_data.iloc[data_row]["BBB+/BBB-"]=="BBB+":
        out.write(raw_data.iloc[data_row]["SMILES"]+"\n")