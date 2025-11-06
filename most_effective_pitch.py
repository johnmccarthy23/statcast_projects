import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 2021 statcast data
statcast21 = pd.read_csv("data/Statcast2021.csv")

# create data with pitch types and zones
# go off of most swinging strikes / least hits

# need des and zone and events
# hit_into_play and field_out






if __name__ == '__main__':
    # strikes are -1
    statcast21.loc[statcast21[statcast21['description'].str.contains('strike')].index, 'launch_speed_angle'] = -1
    # strikeouts are -3
    statcast21.loc[statcast21[statcast21['events'] == 'strikeout'].index, 'launch_speed_angle'] = -3
    # fouls are 0
    statcast21.loc[statcast21[statcast21['description'].str.contains('foul')].index, 'launch_speed_angle'] = 0
    # balls are 1
    statcast21.loc[statcast21[statcast21['description'].str.contains('ball')].index, 'launch_speed_angle'] = 1
    # walks are 3
    statcast21.loc[statcast21[statcast21['events'] == 'walk'].index, 'launch_speed_angle'] = 3

    # print(f"Statcast21: \n{statcast21['launch_speed_angle']}")

    effective_pitches = statcast21[['pitch_type', 'zone', 'p_throws', 'stand', 'launch_speed_angle']].groupby(['pitch_type', 'zone', 'p_throws', 'stand']).sum().reset_index()
    num_pitches = statcast21[['pitch_type', 'zone', 'p_throws', 'stand', 'player_name']].groupby(['pitch_type', 'zone', 'p_throws', 'stand']).count().reset_index()
    # print(f'Num Pitches: \n{num_pitches}')
    # print(f'Grouped Effective Pitches: \n{effective_pitches}')

    effectiveness = effective_pitches['launch_speed_angle'] / num_pitches['player_name']

    effective_pitches['num_pitches'] = num_pitches['player_name']
    effective_pitches['effect_per_pitch'] = effectiveness 
    effective_pitches.rename(columns={'launch_speed_angle': 'effect'}, inplace=True)
    effective_pitches = effective_pitches.loc[effective_pitches[effective_pitches['num_pitches'] >= 100].index, :]

    sorted_effectiveness = effective_pitches.sort_values(by='effect_per_pitch', ascending=True)

    left_left = effective_pitches.loc[effective_pitches.query("p_throws == 'L' & stand == 'L'").index, :]
    left_left_sorted = left_left.sort_values(by='effect_per_pitch', ascending=True)
    
    print(f"Ten Most L-L Effective Pitches: \n{left_left_sorted.head(10)}")

    right_right = effective_pitches.loc[effective_pitches.query("p_throws == 'R' & stand == 'R'").index, :]
    right_right_sorted = right_right.sort_values(by='effect_per_pitch', ascending=True)

    print(f"Ten Most R-R Effective Pitches: \n{right_right_sorted.head(10)}")

    right_left = effective_pitches.loc[effective_pitches.query("p_throws == 'R' & stand == 'L'").index, :]
    right_left_sorted = right_left.sort_values(by='effect_per_pitch', ascending=True)

    print(f"Ten Most R-L Effective Pitches: \n{right_left_sorted.head(10)}")

    left_right = effective_pitches.loc[effective_pitches.query("p_throws == 'L' & stand == 'R'").index, :]
    left_right_sorted = left_right.sort_values(by='effect_per_pitch', ascending=True)

    print(f"Ten Most L-R Effective Pitches: \n{right_left_sorted.head(10)}")


        

        

