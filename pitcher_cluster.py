import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
from sklearn import metrics



# Follow guideline of https://medium.com/swlh/k-means-clustering-on-high-dimensional-data-d2151e1a4240

def data_eda(year, *cols):
        """
        Take the blanket data cleaning steps for all statcast files
        year : year of the statcast data we want to analyze
        cols : the columns we want to initialize our analysis with
        return : a DataFrame consisting of the columns we like for analysis
        """
        # Load 2021 data into a DataFrame
        df = pd.read_csv(f'./data/Statcast{year}.csv')
        print("Full:", df.shape)
        # Drop all deprecated rows
        df.drop(['spin_dir', 'spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated',
                 'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire', 'sv_id'], axis=1, inplace=True)
        
        if len(cols) == 0:
            cols = df.columns

        analysis_cols = list(cols)

        return df[analysis_cols]


def impute_nulls(df, val):
    """ Impute NaN values with some integer
    df : DataFrame we want to impute values into
    val : the integer value we want to impute
    return : the resulting dataframe
    """
    for col in df.columns:
        df.loc[df[col].isnull(), col] = val
    
    return df


def id_to_name(id_df, id):
    """ Take an MLB player's ID and return their name 
    id : a player's id 
    returns : the player's first and last name and team """
    player_name = id_df.loc[id_df["MLBAMID"] == id, "Name"]
    # name = player_name.get("Name")
    if player_name.empty:
        return ''
    else:
        return player_name.iloc[0]
    

        
def main():
    id_df = pd.read_csv("data/player_ids.csv")
    print(f'Test Player Name: {id_to_name(id_df, 594798)}')
    df_2021 = data_eda(2021, 'pitch_type', 'release_speed', 'release_pos_x', 'release_pos_z', 'pitcher',
                             "pfx_x", "pfx_z", "vx0", "vy0", "vz0", "ax",
                             "ay", "az", "effective_speed", "release_spin_rate", "release_extension", "release_pos_y",
                             "spin_axis")

    
    # WHAT WE WANT TO DO IS MAKE SEPARATE ANALYSES FOR EACH PITCH TYPE
    

    # Aggregate by pitcher id
    df_2021 = df_2021.groupby(["pitcher", "pitch_type"]).mean()
    print(df_2021)
    comprehensive_df = df_2021.reset_index()
    print(comprehensive_df)
    df_2021 = impute_nulls(df_2021, 0)

    # Scale the attributes
    scaler = StandardScaler()
    comprehensive_df[df_2021.columns] = scaler.fit_transform(df_2021)
    scaled_df_2021 = comprehensive_df[df_2021.columns]
    print(scaled_df_2021)
    # do PCA to reduce the dimensions of our analysis
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_2021)

    # Show the explained variance percentage
    print('Cumulative variance explained by 2 principal components: {:.2%}'.format(
        np.sum(pca.explained_variance_ratio_)))

    # Results from pca.components_
    dataset_pca = pd.DataFrame(abs(pca.components_), columns=df_2021.columns, index=['PC_1', 'PC_2'])
    print('\n\n', dataset_pca)

    print("\n*************** Most important features *************************")
    print('As per PC 1:\n', (dataset_pca[dataset_pca > 0.3].iloc[0]).dropna())
    print('\n\nAs per PC 2:\n', (dataset_pca[dataset_pca > 0.3].iloc[1]).dropna())
    print("\n******************************************************************")

    # candidate values for our number of cluster
    parameters = [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]
    # instantiating ParameterGrid, pass number of clusters as input
    parameter_grid = ParameterGrid({'n_clusters': parameters})
    best_score = -1
    kmeans_model = KMeans()     # instantiating KMeans model
    silhouette_scores = []
    # evaluation based on silhouette_score
    for p in parameter_grid:
        kmeans_model.set_params(**p)    # set current hyper parameter
        kmeans_model.fit(df_2021)          # fit model on wine dataset, this will find clusters based on parameter p
        ss = metrics.silhouette_score(df_2021, kmeans_model.labels_)   # calculate silhouette_score
        silhouette_scores += [ss]       # store all the scores
        print('Parameter:', p, 'Score', ss)
        # check p which has the best score
        if ss > best_score:
            best_score = ss
            best_grid = p

    # plotting silhouette score
    plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59', width=0.5)
    plt.xticks(range(len(silhouette_scores)), list(parameters))
    plt.title('Silhouette Score', fontweight='bold')
    plt.xlabel('Number of Clusters')
    plt.show()


    # Now we actually fit KMeans
    kmeans = KMeans(n_clusters=5) # we get this from the bar chart
    kmeans.fit(scaled_df_2021)

    names = []
    for id in comprehensive_df["pitcher"]:
        names.append(id_to_name(id_df, id))

    names = np.array(names)
    print(names[:5])

    # Visualize the kmeans clusters
    x = pca_result[:, 0]
    y = pca_result[:, 1]

    fig, ax = plt.subplots(1, figsize=(16, 9))
    sc = ax.scatter(x, y, c=kmeans.labels_, alpha=0.5, s=200)  # plot different colors per cluster
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    # cursor grid lines
    lnx = plt.plot([60,60], [0,1.5], color='black', linewidth=0.3)
    lny = plt.plot([0,100], [1.5,1.5], color='black', linewidth=0.3)
    lnx[0].set_linestyle('None')
    lny[0].set_linestyle('None')

    # annotation
    annot = ax.annotate("", xy=(0,0), xytext=(5,5),textcoords="offset points")
    annot.set_visible(False)

    # xy limits
    plt.xlim(x.min()*0.95, x.max()*1.05)
    plt.ylim(y.min()*0.95, y.max()*1.05)
    def hover(event):
        # check if event was in the axis
        if event.inaxes == ax:
            # draw lines and make sure they're visible
            lnx[0].set_data([event.xdata, event.xdata], [0, 1.5])
            lnx[0].set_linestyle('--')
            lny[0].set_data([0,100], [event.ydata, event.ydata])
            lny[0].set_linestyle('--')
            lnx[0].set_visible(True)
            lny[0].set_visible(True)
            
            # get the points contained in the event
            cont, ind = sc.contains(event)
            if cont:
                # change annotation position
                annot.xy = (-1000, 300)
                # write the name of every point contained in the event
                annot.set_text("{}".format(', '.join([names[n] for n in ind["ind"]])))
                annot.set_visible(True)    
            else:
                annot.set_visible(False)
        else:
            lnx[0].set_visible(False)
            lny[0].set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.title('Pitcher Clusters')
    plt.show()

if __name__ == '__main__':
    main()

