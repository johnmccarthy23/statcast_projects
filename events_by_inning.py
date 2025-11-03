import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2021 statcast data
statcast21 = pd.read_csv("data/Statcast2021.csv")


def group_event_by_inning(df, event):
    statcast_event = df.loc[df[df['events'] == event].index, ['events', 'inning', 'inning_topbot']]
    statcast_grouped = statcast_event.groupby(['inning', 'inning_topbot']).count().reset_index()
    statcast_grouped.rename(columns={'events': event}, inplace=True)
    # only capture 9 innings
    return statcast_grouped[:18]

def plot_event_by_inning(df, event, eventname):
    # split top and bottom
    top_inns = df[df['inning_topbot'] == 'Top']
    bot_inns = df[df['inning_topbot'] == 'Bot']

    innings = np.arange(9) 

    plt.bar(innings - 0.2, top_inns[event], 0.4, label = 'Top Half')
    plt.bar(innings + 0.2, bot_inns[event], 0.4, label = 'Bottom Half')

    plt.xticks(innings, innings+1)
    plt.xlabel("Innings")
    plt.ylabel(f'Number of {eventname}s')
    plt.title(f'Number of {eventname}s per inning')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    k_inn_df = group_event_by_inning(statcast21, 'strikeout')
    plot_event_by_inning(k_inn_df, 'strikeout', 'K')

    hr_inn_df = group_event_by_inning(statcast21, 'home_run')
    plot_event_by_inning(hr_inn_df, 'home_run', 'HR')

    gidp_inn_df = group_event_by_inning(statcast21, 'grounded_into_double_play')
    plot_event_by_inning(gidp_inn_df, 'grounded_into_double_play', 'GIDP')




