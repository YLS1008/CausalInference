from utils import coef_and_bias_list
import numpy as np
import math

def t_learner(df, group):
    ate_list = list()
    c1, b1 = coef_and_bias_list(df.query("T == True"), 'Y', 'Y')
    c0, b0 = coef_and_bias_list(df.query("T == False"), 'Y', 'Y')

    coef_df = df.drop(['Y'], axis=1, inplace=False)
    for index, row in coef_df.iterrows():
        coef_arr = np.asarray(row)
        logit1 = b1 + np.sum(coef_arr * c1)
        logit0 = b0 + np.sum(coef_arr * c0)
        y1_predict = 1 / (1 + math.exp(-logit1))
        y0_predict = 1 / (1 + math.exp(-logit0))
        ate_list.append(y1_predict - y0_predict)

    ate_arr = np.array(ate_list)

    print('The ATE by T-learner on the {} group is: {}'.format(group, ate_arr.mean()))


def IPW(df, group):
    df['ipw1'] = df.apply(lambda x: (x['T'] * x['Y']) / x['e'], axis=1)
    df['ipw2'] = df.apply(lambda x: ((1 - x['T']) * x['Y']) / (1 - x['e']), axis=1)
    print('The ATE by IPW on the {} group is: {}'.format(group, df['ipw1'].mean() - df['ipw2'].mean()))

