
from sklearn.linear_model import LogisticRegression
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


def format_name(s):
    split_list = s.split(" ")
    if len(split_list) > 1:
        new_name = split_list[1] + ', ' + split_list[0]
    else:
        new_name = s
    return new_name


def coef_and_bias_list(df, col_tag, col_drop):
    cov = df.drop(col_drop, axis=1, inplace=False)
    tags = df[col_tag]
    log_reg = LogisticRegression(max_iter=10000)
    log_reg.fit(cov, tags)

    return log_reg.coef_, log_reg.intercept_



def propensity_balance(df, group):
    quintile_list = ['q1', 'q2', 'q3', 'q4', 'q5']
    df['e_quintile'] = pd.cut(x=df['e'], bins=5, labels=quintile_list)
    diff_dict = dict()
    for q in quintile_list:
        treated_df = df.query("T == True & e_quintile == @q")
        control_df = df.query("T == False & e_quintile == @q")

        diff_dict.update({q: 2*(treated_df['e'].mean(axis=0) - control_df['e'].mean(axis=0)) /
                             (treated_df['e'].mean(axis=0) + control_df['e'].mean(axis=0))})

    print('mean difference between control and treatment per quintile for {}:'.format(group))
    print(diff_dict)
    df.drop('e_quintile', axis=1, inplace=True)
