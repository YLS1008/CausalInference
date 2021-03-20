import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RATIO_CLOSE = 0.35
RATIO_MID = 0.26
RATIO_LONG = 0.16

# This file contains all the code used to generate the graphs in the paper.

# Team stats since the year 2000
def offensive_stats_plots():
    stat_list = ["FGA", "FG%", "PTS"]
    avg_df = pd.read_csv('team_avgs.csv')
    plot_df = avg_df[["Season",] + stat_list]

    for stat in stat_list:
        ax = plot_df.plot(x='Season', y=stat)
        plt.ylabel(stat)
        ax.invert_xaxis()
        plt.savefig("graphs/{}_per_game_per_season.png".format(stat))


# histogram of defender's distance from offensive player

def distance_histogram(df, col):
    ax = df.hist(column=col, bins=20)
    ax[0][0].set_xlabel('Feet')
    ax[0][0].set_ylabel('No. of Shots')
    plt.savefig("graphs/{}_distance_histogram.png".format(col))
    plt.clf()


def common_support(df, group):
    bins = np.linspace(0, 1, 100)
    df_treated = df.query("T == True")
    df_control = df.query("T == False")

    plt.hist(df_control['e'], bins, alpha=0.5, label='Control')
    plt.hist(df_treated['e'], bins, alpha=0.5, label='Treated')
    plt.legend(loc='upper right')
    plt.savefig("graphs/common_support_{}.png".format(group))
    plt.clf()



