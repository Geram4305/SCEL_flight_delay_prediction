import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
calendar.setfirstweekday(6)
from datetime import datetime

from parameters import *

class FeatureUtil:
    def __init__(self):
        pass

    @staticmethod   
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def calc_delay_rate(data:pd.DataFrame,groupby_col:str)->pd.DataFrame:
        '''
        Generates delay rate according to given group by column.
        Args:
            data(pd.DataFrame): Dataframe that needs delay_rate calculated.
            groupby_col (str): Column name for group by clause
        Returns:
            data(pd.DataFrame): Datarame with delay_rate calculated
        '''
        res = data.groupby([groupby_col])[DELAY_15_COLUMN].agg(['sum','count'])
        res[DELAY_RATE_COLUMN] = round((res['sum']/res['count'])*100,2)
        return res

    @staticmethod
    def encode_other_top_n(data:pd.DataFrame,col:str,threshold:int=1000)->pd.DataFrame:
        '''
        Encodes 'other' in categorical column based on threshold.
        Args:
            data(pd.DataFrame): Dataframe to be encoded.
            col (str): Column name for encoding
            threshold (int): Minimum no of flights recorded.
        Returns:
            data(pd.DataFrame): Dataframe with 'other' encoded.
        '''
        data.loc[data[col].isin((data[col].value_counts()[data[col].value_counts() < threshold]).index), col] = 'other'
        return data

    @staticmethod
    def flag_flight_change(data:pd.DataFrame)->pd.DataFrame:
        '''
        Flag column for marking flight number change.
        Args:
            data(pd.DataFrame): Dataframe to be encoded.
        Returns:
            data(pd.DataFrame): Dataframe with 'flight_number_change' encoded.
        '''
        data[FLIGHT_NUMBER_CHANGE_COLUMN] = 0
        data.loc[data['Emp-I']!=data['Emp-O'],FLIGHT_NUMBER_CHANGE_COLUMN] = 1
        return data

    @staticmethod
    def flag_destination_change(data:pd.DataFrame)->pd.DataFrame:
        '''
        Flag column for marking destination change.
        Args:
            data(pd.DataFrame): Dataframe to be encoded.
        Returns:
            data(pd.DataFrame): Dataframe with 'dest_change' encoded.
        '''
        data[DEST_CHANGE_COLUMN] = 0
        data.loc[data['Des-I']!=data['Des-O'],DEST_CHANGE_COLUMN] = 1
        return data

    @staticmethod
    def get_week_of_month(dt:datetime.date)->int:
        '''
        Calculates week of the month given date.
        Args:
            dt(datetime.date): Date.
        Returns:
            week_of_month(int): week of the month.
        '''
        year, month, day = dt.year, dt.month, dt.day
        x = np.array(calendar.monthcalendar(year, month))
        week_of_month = np.where(x==day)[0][0] + 1
        return week_of_month

    @staticmethod
    def get_no_of_flights_same_day(data:pd.DataFrame)->pd.DataFrame:
        '''
        Calculates no of flights taking off on a given date.
        Args:
            data(pd.DataFrame): Dataframe that needs same_day_flight calculated.
        Returns:
            merged(pd.DataFrame): Dataframe with same_day_flight calculated.
        '''
        data_grpby = data.groupby(['MES','DIA'])['Fecha-I'].agg(['count']).reset_index()
        merged = data.merge(data_grpby, on=['MES','DIA'])
        merged.rename(columns={"count": SAME_DAY_FLIGHTS},inplace=True)
        return merged

    @staticmethod
    def drop_cols(data:pd.DataFrame,drop_cols:list=[])->pd.DataFrame:
        '''
        Drops the given list of columns.
        Args:
            data(pd.DataFrame): Dataframe.
            drop_cols(list): Cols list to drop
        Returns:
            data(pd.DataFrame): Dataframe with cols dropped
        '''
        return data.drop(columns=drop_cols)

class PlotUtil:
    def __init__(self):
        pass

    @staticmethod
    def plot_bar_delay_rate(data:pd.DataFrame,groupby_col:str,bar_col:str='count')->object:
        '''
        Generates bar plot with delay rate and count of flights according to given group by column.
        Args:
            data(pd.DataFrame): Dataframe with delay_rate calculated.
            groupby_col (str): Column name for group by clause
            bar_col (str): Column name for bars in the bar plot
        Returns:
            fig(plt.figure): Matplotlib figure
        '''
        fig = plt.figure(figsize=(21,9)) # Create matplotlib figure
        ax = fig.add_subplot(111) # Create matplotlib axes
        ax2 = ax.twinx()
        width = 0.2
        ax.set_title('Delay rate wrt {}'.format(groupby_col))
        data[bar_col].plot(kind='bar', color='red', ax=ax,label='No of flights operated')
        data[DELAY_RATE_COLUMN].plot(color='blue', ax=ax2, label='Delay rate (%)')
        plt.axhline(data[DELAY_RATE_COLUMN].mean(), color='g', linestyle='dashed', linewidth=2,label='Mean delay rate')
        rotation = 0
        if(groupby_col in ['SIGLADES','OPERA']):
            rotation = 90
        ax.tick_params(axis='x', rotation=rotation)
        ax.set_ylabel('No of flights operated')
        ax.legend(loc='upper left')
        ax2.set_ylabel('Delay rate (%)')
        ax2.set_ylim([0, 100])
        ax2.legend(loc='upper right')
        return fig