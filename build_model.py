import numpy as np
import pandas as pd
from scrape_data import *

class LinearModel:

	def __init__(self, dataframe):
		self.data = dataframe

	def getStatistics(self, column_name = ''):
		column = self.data[column_name].values
		mean = np.mean(column)
		variance = np.var(column)
		return mean, variance

	def buildLinearModel(self, variables_dict, model_type = 'linear_regression'):
		pass

	def createTeamVariable(self, column_name):
		group_data = {}
		for team in self.data.groupby(by = ['team']).groups.keys():
			group_data[team] = self.data.groupby(by = ['team']).get_group(team)

		group_statistic = []
		for i in range(len(self.data)):
			player_team = self.data.iloc[i]['team']
			team_data_without_player = group_data[player_team][group_data[player_team]['player'] != self.data.iloc[i]['player']]
			group_statistic.append(np.mean(team_data_without_player[column_name].values))

		self.data['rest_of_team_{}'.format(column_name)] = pd.Series(group_statistic)

		# print(self.data.groupby(by = ['team']).get_group('Manchester City'))

		

if __name__ == '__main__':
	dataframe = pd.read_csv('/home/parth/Projects/UCSD/ProbabilityAndStatistics/Project/PL2021_Outfield.csv')
	# variables_dict = {'dependent': 'goals_per90','independent_player': ''}
	L = LinearModel(dataframe)

	L.createTeamVariable('assists')