import numpy as np
import pandas as pd
from scrape_data import *
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class DataGenerator:

	def __init__(self, dataframe1, dataframe2=None):
		self.data = dataframe1
		self.data = self.data.add_prefix('club_')

		if dataframe2 is not None:
			self.data2 = dataframe2
			self.data2 = self.data2.add_prefix('international_')

		self.merged = pd.merge(self.data, self.data2, how = 'inner', left_on = 'club_player', right_on='international_player')

	def getStatistics(self, column_name = ''):
		column = self.data[column_name].values
		mean = np.mean(column)
		variance = np.var(column)
		return mean, variance

	def plotDistribution(self, column_name = ''):
		column = self.merged[column_name].values
		plt.hist(column, bins=30)
		plt.show()

	def createTeamVariable(self, column_name):
		group_data = {}
		for team in self.merged.groupby(by = ['club_team']).groups.keys():
			group_data[team] = self.merged.groupby(by = ['club_team']).get_group(team)

		group_statistic = []
		cardinality = []
		for i in range(len(self.merged)):
			player_team = self.merged.iloc[i]['club_team']
			team_data_without_player = group_data[player_team][group_data[player_team]['club_player'] != self.merged.iloc[i]['club_player']]
			cardinality.append(len(group_data[player_team][group_data[player_team]['club_player'] != self.merged.iloc[i]['club_player']]))
			group_statistic.append(np.nanmean(team_data_without_player['club_' + column_name].values))

		self.merged['rest_of_club_team_{}'.format('club_' + column_name)] = pd.Series(group_statistic)
		self.merged['rest_of_club_team_{}'.format('cardinality')] = pd.Series(cardinality)

		# group_data = {}
		# for team in self.merged.groupby(by = ['international_team']).groups.keys():
		# 	group_data[team] = self.merged.groupby(by = ['international_team']).get_group(team)

		# group_statistic = []
		# cardinality = []
		# for i in range(len(self.merged)):
		# 	player_team = self.merged.iloc[i]['international_team']
		# 	team_data_without_player = group_data[player_team][group_data[player_team]['international_player'] != self.merged.iloc[i]['international_player']]
		# 	cardinality.append(len(group_data[player_team][group_data[player_team]['club_player'] != self.merged.iloc[i]['club_player']]))
		# 	group_statistic.append(np.nanmean(team_data_without_player['club_' + column_name].values))

		# self.merged['rest_of_international_team_{}'.format('club_' + column_name)] = pd.Series(group_statistic)
		# self.merged['rest_of_international_team_{}'.format('cardinality')] = pd.Series(cardinality)

	def createCombinedDataframe(self):
		return pd.merge(self.data, self.data2, left_on = 'club_player', right_on='international_player')

class LinearModel:

	def __init__(self, dataframe):
		self.data = dataframe

	def buildLinearModel(self, variables_dict, model_type = 'linear_regression'):
		y = self.data[variables_dict['dependent']].values
		X = self.data.select_dtypes(include=np.number).apply(lambda x: x.fillna(x.mean()),axis=0).drop(['international_xg_per90', 'international_goals_per90'], axis=1)

		positions_list = ['FW']

		# print(self.data['club_position'].values in positions_list)

		# y = y[(self.data['rest_of_club_team_cardinality'].values > 8)]
		# X = X[(self.data['rest_of_club_team_cardinality'].values > 8)]

		print(len(X))

		self.reg = LinearRegression().fit(X, y)

		print(self.reg.score(X, y))

	def evaluateLinearModel(self, variables_dict):
		y = self.data[variables_dict['dependent']].values
		X = self.data[variables_dict['independent']].values
		
		y = y[self.data['rest_of_international_team_cardinality'].values > 8]
		X = X[self.data['rest_of_international_team_cardinality'].values > 8]

		print(self.reg.score(X, y))

if __name__ == '__main__':
	dataframe = pd.read_csv('/home/parth/Projects/UCSD/ProbabilityAndStatistics/Project/data/Big52020-21_Outfield.csv')
	dataframe2 = pd.read_csv('/home/parth/Projects/UCSD/ProbabilityAndStatistics/Project/data/Euro2021_Outfield.csv')
	L = DataGenerator(dataframe, dataframe2)

	L.createTeamVariable('xg_assist_per90')
	# L.createTeamVariable('goals_per90')
	# L.createTeamVariable('xg_xg_assist_per90')
	# L.createTeamVariable('passes_pct')
	# L.createTeamVariable('passes_completed')
	L.createTeamVariable('sca_per90')

	L.merged['club_goals_minus_xg_per90'] = L.merged['club_goals_per90'] - L.merged['club_xg_per90']
	L.merged['club_touches_att_3rd_per90'] = L.merged['club_touches_att_3rd']*90/L.merged['club_minutes']
	L.merged.to_csv('Euro21_PL21_corrected.csv')

	fit_variables_dict = {'dependent': 'international_xg_per90',
	'independent': ['club_sca_per90']}

	M = LinearModel(L.merged)
	M.buildLinearModel(fit_variables_dict)

	# predict_variables_dict = {'dependent': 'international_goals_per90',
	# 'independent': ['rest_of_international_team_club_xg_assist_per90', 
	# 			    'rest_of_international_team_club_sca_per90']}

	# M.evaluateLinearModel(predict_variables_dict)

	L.plotDistribution('club_touches_att_3rd_per90')