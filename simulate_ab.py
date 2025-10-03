import pandas as pd
import matplotlib.pyplot as plt

# Read in 2021 file
statcast21 = pd.read_csv("../Statcast_2021.csv")

# get pitcher data
pitcher_data = statcast21[statcast21[""]]