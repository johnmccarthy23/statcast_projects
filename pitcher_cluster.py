import pandas as pd

def data_eda(year, *cols):
        """
        Take the blanket data cleaning steps for all statcast files
        year : year of the statcast data we want to analyze
        cols : the columns we want to initialize our analysis with
        """
        # Load 2021 data into a DataFrame
        df = pd.read_csv(f'statcast_{year}.csv')

        # Drop all deprecated rows
        df.drop(['spin_dir', 'spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated', 'tfs_deprecated', 
        'tfs_zulu_deprecated', 'umpire', 'sv_id'], axis=1, inplace=True)
        
        if len(cols) == 0:
            cols = df.columns

        analysis_cols = list(cols)

        return df[analysis_cols]



        
def main():
    df_2021 = data_eda(2021, 'pitch_type', 'release_speed', 'release_pos_x', 'release_pos_z', 'pitcher',
                              'p_throws', "pfx_x", "pfx_z", "vx0", "vy0", "vz0", "ax", 
                              "ay", "az", "effective_speed", "release_spin_rate", "release_extension", "release_pos_y", 
                              "spin_axis")

    df_2021 = df_2021.groupby(["pitcher", "pitch_type"]).mean()


if __name__ == '__main__':
    main()