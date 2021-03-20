from utils import format_name, coef_and_bias_list

import pandas as pd
import numpy as np
import math

SPLIT = 0.2 # the top percentage taken as quality defender



def basic_data_formatting():
    # read relevant columns from csv and filter out players with under 10 games played
    col_list1 = ['Player', 'G', 'DRtg']
    col_list2 = ['SHOT_NUMBER', 'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST', 'SHOT_RESULT',
                    'CLOSEST_DEFENDER', 'CLOSE_DEF_DIST', 'FGM', 'PTS']
    per100_df = pd.read_csv('per100.csv', usecols=col_list1).astype({ 'G': 'int32', 'DRtg': 'int32'})
    per100_df.query('G > 9', inplace=True)

    # reformat player name to match the shot logs
    per100_df['Player'] = per100_df['Player'].apply(lambda x: "{}".format(x.split('\\')[0]))
    per100_df['Player'] = per100_df['Player'].apply(lambda x: format_name(x))

    # sort by Defensive Rating and make a 20 / 80 split
    per100_df.sort_values(by='DRtg', inplace=True)
    split_index = int(len(per100_df.index) * SPLIT)
    quality_df = per100_df.head(split_index).copy()
    regular_df = per100_df.tail(len(per100_df.index) - split_index).copy()

    # tag players by defensive rating, concat and create a dictionary of tags
    quality_df['Q_defender'] = 1
    regular_df['Q_defender'] = 0
    tagged_players_df = pd.concat([quality_df, regular_df]) #
    tagged_players_df.drop(['G', 'DRtg'], axis='columns', inplace=True)
    tagged_dict = dict(tagged_players_df.values)

    # read shot logs and add tags
    shotlogs_df = pd.read_csv('shot_logs.csv', usecols=col_list2)
    shotlogs_df['T'] = shotlogs_df['CLOSEST_DEFENDER'].map(tagged_dict)
    shotlogs_df['T'].fillna(value=0, inplace=True)
    shotlogs_df.rename(columns={'SHOT_RESULT': 'Y'}, inplace=True)
    shotlogs_df['Y'] = shotlogs_df['Y'].apply(lambda x: 1 if x == 'made' else 0)
    shotlogs_df['FG_PERCENTAGE'] = shotlogs_df.apply(lambda x: x['FGM']/x['SHOT_NUMBER'], axis=1)
    shotlogs_df.drop(['CLOSEST_DEFENDER', 'FGM', 'SHOT_NUMBER'], axis='columns', inplace=True)

    return shotlogs_df

def data_filtering(df, dist):
    df.drop(df[df.CLOSE_DEF_DIST > dist].index, inplace=True)
    return df


def categorical_formatting(df):
    df['SHOT_DIST'] = pd.cut(x=df['SHOT_DIST'], bins=[0, 10, 20, 40], labels=['close', 'mid', 'long'])
    df['SHOT_CLOCK'] = pd.cut(x=df['SHOT_CLOCK'], bins=[0, 21, 24], labels=[0, 1])
    df['SHOT_CLOCK'].fillna(value=1, inplace=True)
    df_close = df.query("SHOT_DIST == 'close'")
    df_mid = df.query("SHOT_DIST == 'mid'")
    df_long = df.query("SHOT_DIST == 'long'")

    df_close.drop(['SHOT_DIST'], axis='columns', inplace=True)
    df_mid.drop(['SHOT_DIST'], axis='columns', inplace=True)
    df_long.drop(['SHOT_DIST'], axis='columns', inplace=True)

    return df_close, df_mid, df_long

def write_propensity_score(df_list):
    for df in df_list:
        c, b = coef_and_bias_list(df, 'T', ['T', 'Y'])
        propensity_list = list()
        coef_df = df.drop(['T', 'Y'], axis=1, inplace=False)
        for index, row in coef_df.iterrows():
                coef_arr = np.asarray(row)
                logit = b + np.sum(coef_arr * c)
                propensity_list.append(1/(1 + math.exp(-logit)))

        df['e'] = propensity_list




