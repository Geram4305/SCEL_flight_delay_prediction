import pandas as pd
import matplotlib.pyplot as plt

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
    def calc_delay_rate(data:pd.DataFrame,groupby_col:str):
        '''
        Generates delay rate according to given group by column.
        Args:
            data(pd.DataFrame): Dataframe that needs delay_rate calculated.
            groupby_col (str): Column name for group by clause
        Returns:
            data(pd.DataFrame): Datarame with delay_rate calculated
        '''
        res = data.groupby([groupby_col])['delay_15'].agg(['sum','count'])
        res[DELAY_RATE_COLUMN] = round((res['sum']/res['count'])*100,2)
        return res

class PlotUtil:
    def __init__(self):
        pass

    def plot_bar_delay_rate(data:pd.DataFrame,groupby_col:str,bar_col:str='count'):
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