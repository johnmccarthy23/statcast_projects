import pandas as pd
import matplotlib.pyplot as plt

def name_to_id(id_df, name):
    player_id = id_df.loc[id_df["Name"] == name, "MLBAMID"]
    # name = player_name.get("Name")
    if player_id.empty:
        return ''
    else:
        return player_id.iloc[0]

# Read in 2021 file
statcast21 = pd.read_csv("../Statcast_2021.csv")



# get pitcher data
pitcher_data = statcast21[statcast21[""]]