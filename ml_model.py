import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.metrics import f1_score,roc_auc_score,roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from utilities import *


class MLModel:
	def __init__(self,data:pd.DataFrame,target_col:str,seed=42)->None:
		self.seed = seed
		self.data = data
		self.target_col = target_col
		self.models = {'random_forest':RandomForestClassifier(random_state=self.seed,verbose=1,class_weight='balanced_subsample',n_jobs=-1),
		'log_regression':LogisticRegression(random_state=self.seed,verbose=1,class_weight='balanced',n_jobs=-1)}
		self.param_grids = {
		'random_forest':{ 
					    'n_estimators': [50,100,150],
					    'max_depth' : [4,5,6,7,8],
					    'criterion' :['gini', 'entropy']},
		'log_regression':{ 
						 'penalty': ['l2','elasticnet','l1'],
						 'max_iter' : [100,50,150,200]
						 }
						    }
		self.x_test,self.y_test = None, None
		self.feat_cols = None

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

	def __encode_categorical_cols(self)->list:
		self.data = pd.get_dummies(self.data,drop_first=True)
		return self.data.columns.values

	def create_synthetic_features(self)->None:
		self.data = FeatureUtil.flag_flight_change(self.data)
		self.data = FeatureUtil.flag_destination_change(self.data)
		self.data[WEEK_OF_MONTH_COLUMN] = self.data['Fecha-I'].dt.date.apply(FeatureUtil.get_week_of_month)
		self.data = FeatureUtil.get_no_of_flights_same_day(self.data)

	def __normalize_features(self):
		self.data = MinMaxScaler().fit_transform(self.data)

	def choose_feature_list(self,exclude:list=[])->None:
		self.data = FeatureUtil.drop_cols(self.data,exclude)

	def train(self)->object:
		self.__convert_to_categorical()
		self.data,self.target = self.__get_feature_target_data(self.data,self.target_col)
		self.feat_cols = self.__encode_categorical_cols()
		self.__normalize_features()

		#create test datasets for further testing or fine tuning of thresholds.
		self.data, self.x_test, self.target, self.y_test = train_test_split(self.data, self.target, test_size=0.2, random_state=self.seed, stratify=self.target)

		results = {}
		for model_key in self.models.keys():
			results[model_key] = GridSearchCV(estimator=self.models[model_key], param_grid=self.param_grids[model_key], cv= 3,\
			scoring=['roc_auc','f1'], refit='roc_auc',verbose=1,n_jobs=-1,return_train_score=True).fit(self.data, self.target)

		return results

	def predict(self,x_test:list=[])->list:
		return self.model.predict(x_test)


class MLUtil:
	def __init__(self,test_size:float=0.33,seed:int=42)->None:
		#self.thresholds = np.arange(0, 1, 0.001)
		pass

	def __to_labels(self,pos_probs:float, threshold:float)->int:
		return (pos_probs >= threshold).astype('int')

	def tune_threshold_log_regression(self,model:object,x_test:pd.DataFrame,y_test:pd.DataFrame)->Tuple[float,float]:
		# predict probabilities
		y_pred = model.predict_proba(x_test)
		# keep probabilities for the positive outcome only
		probs = y_pred[:, 1]

		fpr, tpr, thresholds = roc_curve(y_test, probs)
		gmeans = np.sqrt(tpr * (1-fpr))
		# # evaluate each threshold
		# f1_scores = [f1_score(y_test, self.__to_labels(probs, t)) for t in self.thresholds]

		# roc_auc_scores = [roc_auc_score(y_test, self.__to_labels(probs, t)) for t in self.thresholds]
		# # get best threshold
		ind = np.argmax(gmeans)

		return thresholds[ind], roc_auc_score(y_test, self.__to_labels(probs, thresholds[ind]))\
		,f1_score(y_test, self.__to_labels(probs, thresholds[ind]))