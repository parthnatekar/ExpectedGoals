import numpy as np
import pandas as pd
from scrape_data import *
from sklearn.linear_model import LinearRegression

class DataGenerator:

	def __init__(self, dataframe1, dataframe2=None):
		self.data = dataframe1
		self.data = self.data.add_prefix('club_')

		if dataframe2 is not None:
			self.data2 = dataframe2
			self.data2 = self.data2.add_prefix('international_')

	def getStatistics(self, column_name = ''):
		column = self.data[column_name].values
		mean = np.mean(column)
		variance = np.var(column)
		return mean, variance

	def createTeamVariable(self, column_name):
		group_data = {}
		for team in self.data.groupby(by = ['club_team']).groups.keys():
			group_data[team] = self.data.groupby(by = ['club_team']).get_group(team)

		group_statistic = []
		for i in range(len(self.data)):
			player_team = self.data.iloc[i]['club_team']
			team_data_without_player = group_data[player_team][group_data[player_team]['club_player'] != self.data.iloc[i]['club_player']]
			group_statistic.append(np.mean(team_data_without_player['club_' + column_name].values))

		self.data['rest_of_team_{}'.format('club_' + column_name)] = pd.Series(group_statistic)

		group_data = {}
		for team in self.data2.groupby(by = ['international_team']).groups.keys():
			group_data[team] = self.data2.groupby(by = ['international_team']).get_group(team)

		group_statistic = []
		for i in range(len(self.data2)):
			player_team = self.data2.iloc[i]['international_team']
			team_data_without_player = group_data[player_team][group_data[player_team]['international_player'] != self.data2.iloc[i]['international_player']]
			group_statistic.append(np.mean(team_data_without_player['international_' + column_name].values))

		self.data2['rest_of_team_{}'.format('international_' + column_name)] = pd.Series(group_statistic)

	def createCombinedDataframe(self):
		return pd.merge(self.data, self.data2, left_on = 'club_player', right_on='international_player')

class LinearModel:

	def __init__(self, dataframe):
		self.data = dataframe

	def buildLinearModel(self, variables_dict, model_type = 'linear_regression'):
		y = self.data[variables_dict['dependent']]
		X = self.data[variables_dict['independent']]

		reg = LinearRegression().fit(X, y)

		print(reg.score(X, y), reg.coef_)

if __name__ == '__main__':
	dataframe = pd.read_csv('/home/parth/Projects/UCSD/ProbabilityAndStatistics/Project/data/PL2020-21_Outfield.csv')
	dataframe2 = pd.read_csv('/home/parth/Projects/UCSD/ProbabilityAndStatistics/Project/data/Euro2021_Outfield.csv')
	L = DataGenerator(dataframe, dataframe2)

	L.createTeamVariable('assists_per90')
	L.createTeamVariable('goals_per90')
	L.createTeamVariable('xg_xg_assist_per90')
	L.createTeamVariable('passes_pct')
	L.createTeamVariable('passes_completed')

	merged = L.createCombinedDataframe()
	merged.to_csv('Euro21_PL21.csv')

	variables_dict = {'dependent': 'international_goals_per90',
	'independent': ['club_goals_per90', 'rest_of_team_international_assists_per90', 'rest_of_team_international_xg_xg_assist_per90']}

	M = LinearModel(merged)
	M.buildLinearModel(variables_dict)