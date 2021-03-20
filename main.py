AFFECTIVE_DIST = 8 # in feet


import data_formatting as dtf
import graphs_and_plots as gnp
import causal_algorithms as alg
from utils import propensity_balance


def main():
    gnp.offensive_stats_plots()
    raw_df = dtf.basic_data_formatting()
    # gnp.distance_histogram(raw_df, 'CLOSE_DEF_DIST')
    # gnp.distance_histogram(raw_df, 'SHOT_DIST')
    filtered_df = dtf.data_filtering(raw_df, AFFECTIVE_DIST)

    close_df, mid_df, long_df = dtf.categorical_formatting(filtered_df)
    dtf.write_propensity_score([close_df, mid_df, long_df])
    groups_list = [(close_df, 'close'), (mid_df, 'mid'), (long_df, 'long')]

    for curr in groups_list:
        (df, group) = curr
        # gnp.common_support(df, group)
        propensity_balance(df, group)
        alg.t_learner(df, group)
        alg.IPW(df, group)




if __name__ == "__main__":
    main()