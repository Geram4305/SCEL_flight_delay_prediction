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

def calc_delay15(data:pd.DataFrame)->pd.DataFrame:
    '''
    Generates delay_15 column.
    Args:
        data(pd.DataFrame): Dataframe that needs delay_15 calculated.
    Returns:
        data(pd.DataFrame): Datarame with delay_15 column calculated
    '''
    data[DELAY_15_COLUMN] = 0
    data.loc[data[MINUTES_DIFF_COLUMN]>15,DELAY_15_COLUMN] = 1
    return data

def calc_period_day(hour:int)->str:
    '''
    Generates period of day from hour of day.
    Args:
        hour(int): Hour of day
    Returns:
        str: Period of day - Morning, Afternoon or Night
    '''
    if (hour >=5) and (hour < 12):
        return 'Morning'
    elif (hour >= 12) and (hour < 19 ):
        return 'Afteroon'
    elif (hour >= 19) or (hour < 5):
        return'Night'