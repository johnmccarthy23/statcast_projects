import pandas as pd
import matplotlib.pyplot as plt

# Read in 2021 file
statcast21 = pd.read_csv("Statcast_2021.csv")


# Columns and meanings here -- https://baseballsavant.mlb.com/csv-docs


def launch_hit():
    # Plot launch angle and hit speed by hit type
    # Only use actual hits [single, double, triple, HR]

    singles = statcast21.loc[statcast21['events'] == 'single', ['launch_speed', 'launch_angle', 'events']]
    doubles = statcast21.loc[statcast21['events'] == 'double', ['launch_speed', 'launch_angle', 'events']]
    triples = statcast21.loc[statcast21['events'] == 'triple', ['launch_speed', 'launch_angle', 'events']]
    homers = statcast21.loc[statcast21['events'] == 'home_run', ['launch_speed', 'launch_angle', 'events']]


    # Plot launch angle and hit speed by hit type
    plt.scatter(singles['launch_speed'], singles['launch_angle'], c = 'blue', label = 'single')
    plt.scatter(doubles['launch_speed'], doubles['launch_angle'], c = 'green', label = 'double')
    plt.scatter(triples['launch_speed'], triples['launch_angle'], c = 'yellow', label = 'triple')
    plt.scatter(homers['launch_speed'], homers['launch_angle'], c = 'red', label = 'home run')
    plt.legend()
    plt.xlabel('Launch Speed')
    plt.ylabel('Launch Angle')
    plt.title("Hit Type by Launch Speed and Launch Angle")
    plt.show()

if __name__ == '__main__':
    launch_hit()
