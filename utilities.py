import pandas as pd
from parameters import *

def calc_high_season(data:pd.DataFrame)->pd.DataFrame:
    '''
    Generates high_season column. 1 when True and 0 otherwise.
    Args:
        data(pd.DataFrame): Dataframe that needs high season calculated.
    Returns:
        data(pd.DataFrame): Datarame with high_season column calculated
    '''
    data[HIGH_SEASON_COLUMN] = 0
    data.loc[(data[TIMESTAMP_SCHEDULED_COLUMN] >= '2017-07-15') & (data[TIMESTAMP_SCHEDULED_COLUMN] <= '2017-07-31'),HIGH_SEASON_COLUMN] = 1
    data.loc[(data[TIMESTAMP_SCHEDULED_COLUMN] >= '2017-12-15') & (data[TIMESTAMP_SCHEDULED_COLUMN] <= '2017-12-31'),HIGH_SEASON_COLUMN] = 1
    data.loc[(data[TIMESTAMP_SCHEDULED_COLUMN] >= '2017-01-01') & (data[TIMESTAMP_SCHEDULED_COLUMN] <= '2017-03-03'),HIGH_SEASON_COLUMN] = 1
    data.loc[(data[TIMESTAMP_SCHEDULED_COLUMN] >= '2017-09-11') & (data[TIMESTAMP_SCHEDULED_COLUMN] <= '2017-09-30'),HIGH_SEASON_COLUMN] = 1
    return data

def calc_min_diff(data:pd.DataFrame)->pd.DataFrame:
    '''
    Generates min_diff column.
    Args:
        data(pd.DataFrame): Dataframe that needs min_diff calculated.
    Returns:
        data(pd.DataFrame): Datarame with min_diff column calculated
    '''
    data[MINUTES_DIFF_COLUMN] = (data[TIMESTAMP_OPERATING_COLUMN] - data[TIMESTAMP_SCHEDULED_COLUMN]) / pd.Timedelta(minutes=1)
    return data