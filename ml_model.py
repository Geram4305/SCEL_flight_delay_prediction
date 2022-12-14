import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.metrics import f1_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score

from utilities import *


class MLModel:
	def __init__(self,data:pd.DataFrame,target_col:str,seed=42)->None:
		self.seed = seed
		self.data = data
		self.target_col = target_col
		self.model = RandomForestClassifier(random_state=self.seed,verbose=1,class_weight='balanced_subsample',n_jobs=-1)

	def __convert_to_categorical(self)->None:
		cols=self.data.select_dtypes('object').columns.values
		self.data[cols]=self.data[cols].astype('category')

	def __get_feature_target_data(self,df:pd.DataFrame,target_col:str)->Tuple[pd.DataFrame,pd.DataFrame]:
		target = df[target_col]
		df = df.drop(columns=[target_col])
		df = df.reset_index(drop=True)
		return df, target

	def __train_test_split(self):
		return train_test_split(self.data, self.target, test_size=0.33, random_state=self.seed, stratify=self.target)

	def encode_cols_as_other(self,encode_cols:list=[])->None:
		for col in encode_cols:
			self.data = FeatureUtil.encode_other_top_n(self.data,col)

	def __encode_categorical_cols(self)->None:
		self.data = pd.get_dummies(self.data,drop_first=True)

	def create_synthetic_features(self)->None:
		self.data = FeatureUtil.flag_flight_change(self.data)
		self.data = FeatureUtil.flag_destination_change(self.data)
		self.data[WEEK_OF_MONTH_COLUMN] = self.data['Fecha-I'].dt.date.apply(FeatureUtil.get_week_of_month)
		self.data = FeatureUtil.get_no_of_flights_same_day(self.data)

	def choose_feature_list(self,exclude:list=[])->None:
		self.data = FeatureUtil.drop_cols(self.data,exclude)

	def train(self)->Tuple[float,float]:
		self.__convert_to_categorical()
		self.data,self.target = self.__get_feature_target_data(self.data,self.target_col)
		self.__encode_categorical_cols()
		return cross_val_score(self.model,self.data.values, self.target.values,n_jobs=-1,verbose=1,scoring='f1',cv=5)

	def predict(self,x_test:list=[])->list:
		return self.model.predict(x_test)

