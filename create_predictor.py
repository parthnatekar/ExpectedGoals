import numpy as np
import pandas as pd
from scrape_data import *
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.stats as ss
from scipy.optimize import curve_fit
from scipy.stats import poisson
import matplotlib.mlab as mlab
import os
from collections import defaultdict
import itertools
from functools import reduce

class WorldCupPredictor:

	def __init__(self, data_path):
		self.data_path = data_path
		region_weight = {'AFC': 0.5, 'UEFA': 0.75, 'CONCACAF': 0.5, 'CONMEBOL': 1.5, 'CAF': 0.5}
		for file in os.listdir(self.data_path):
			region_data = pd.read_csv(self.data_path + '/' + file, header=1)
			region_data['Gls.1'] *= region_weight[file.split('.')[0]]
			try: 
				self.data = pd.concat([self.data, region_data], ignore_index=True)
			except:
				self.data = region_data

		self.groups = {'A': ['Netherlands', 'Senegal', 'Ecuador', 'Qatar'],
					   'B': ['England', 'United States', 'IR Iran', 'Wales'],
					   'C': ['Argentina', 'Poland', 'Mexico', 'Saudi Arabia'],
					   'D': ['France', 'Australia', ' Tunisia', 'Denmark'],
					   'E': ['Japan', 'Spain', 'Germany', 'Costa Rica'],
					   'F': ['Morocco', 'Croatia', 'Belgium', 'Canada'],
					   'G': ['Brazil', 'Switzerland', 'Cameroon', 'Serbia'],
					   'H': ['Portugal', 'Korea Republic', 'Uruguay', 'Ghana']}

	def computeExpectedTeamGoals(self):
		self.expectedGoals = defaultdict(list)

		for group in self.groups:
			for team in self.groups[group]:
				self.expectedGoals[team] = self.data.loc[self.data['Squad'].str.contains(team)]['Gls.1'].values[0]
		print(self.expectedGoals)

	def predictGroupStages(self):
		self.groupPredictions = {group:{team:np.array([0, 0]) for team in self.groups[group]} for group in self.groups}

		for group in self.groups:
			for item in list(itertools.combinations(self.groups[group], 2)):
				eG1 = poisson.rvs(self.expectedGoals[item[0]])
				eG2 = poisson.rvs(self.expectedGoals[item[1]])
				if eG1 > eG2:
					self.groupPredictions[group][item[0]] += np.array([3, eG1 - eG2])
					self.groupPredictions[group][item[1]] += np.array([0, -eG1 + eG2])
				elif eG1 < eG2:
					self.groupPredictions[group][item[1]] += np.array([3, eG1 - eG2])
					self.groupPredictions[group][item[1]] += np.array([0, -eG1 + eG2])
				else:
					self.groupPredictions[group][item[0]] += np.array([1, 0])
					self.groupPredictions[group][item[1]] += np.array([1, 0])

		for g in self.groupPredictions:
			self.groupPredictions[g] = {k: v for k, v in sorted(self.groupPredictions[g].items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)}

		self.groupWinners = {group:[team for team, value in list(self.groupPredictions[group].items())[:2]] for group in self.groupPredictions}
	
	def createKnockoutTree(self):
		# self.knockoutTree = [[[[self.groupWinners['A'][0], self.groupWinners['B'][1]],
		# 					   [self.groupWinners['C'][0], self.groupWinners['D'][1]]],

		# 					[[[self.groupWinners['E'][0], self.groupWinners['F'][1]],
		# 					  [self.groupWinners['G'][0], self.groupWinners['H'][1]]]]],

		# 					[[[self.groupWinners['A'][1], self.groupWinners['B'][0]],
		# 					  [self.groupWinners['C'][1], self.groupWinners['D'][0]]],

		# 					[[[self.groupWinners['E'][1], self.groupWinners['F'][0]],
		# 					  [self.groupWinners['G'][1], self.groupWinners['H'][0]]]]]]
		
		def add_element(root, path, data):
			reduce(lambda x, y: x[y], path[:-1], root)[path[-1]] = data

		tree = lambda: defaultdict(tree)
		self.knockoutTree = tree()

		add_element(self.knockoutTree, ['Semi-Final-1', 'Quarter-Final-1', 'Pre-Quarter-1', 'Bracket-1'], self.groupWinners['A'][0])
		add_element(self.knockoutTree, ['Semi-Final-1', 'Quarter-Final-1', 'Pre-Quarter-1', 'Bracket-2'], self.groupWinners['B'][1])
		add_element(self.knockoutTree, ['Semi-Final-1', 'Quarter-Final-1', 'Pre-Quarter-2', 'Bracket-3'], self.groupWinners['C'][0])
		add_element(self.knockoutTree, ['Semi-Final-1', 'Quarter-Final-1', 'Pre-Quarter-2', 'Bracket-4'], self.groupWinners['D'][1])
		add_element(self.knockoutTree, ['Semi-Final-1', 'Quarter-Final-2', 'Pre-Quarter-3', 'Bracket-5'], self.groupWinners['E'][0])
		add_element(self.knockoutTree, ['Semi-Final-1', 'Quarter-Final-2', 'Pre-Quarter-3', 'Bracket-6'], self.groupWinners['F'][1])
		add_element(self.knockoutTree, ['Semi-Final-1', 'Quarter-Final-2', 'Pre-Quarter-4', 'Bracket-7'], self.groupWinners['G'][0])
		add_element(self.knockoutTree, ['Semi-Final-1', 'Quarter-Final-2', 'Pre-Quarter-4', 'Bracket-8'], self.groupWinners['H'][1])
		add_element(self.knockoutTree, ['Semi-Final-2', 'Quarter-Final-3', 'Pre-Quarter-5', 'Bracket-9'], self.groupWinners['A'][1])
		add_element(self.knockoutTree, ['Semi-Final-2', 'Quarter-Final-3', 'Pre-Quarter-5', 'Bracket-10'], self.groupWinners['B'][0])
		add_element(self.knockoutTree, ['Semi-Final-2', 'Quarter-Final-3', 'Pre-Quarter-6', 'Bracket-11'], self.groupWinners['C'][1])
		add_element(self.knockoutTree, ['Semi-Final-2', 'Quarter-Final-3', 'Pre-Quarter-6', 'Bracket-12'], self.groupWinners['D'][0])
		add_element(self.knockoutTree, ['Semi-Final-2', 'Quarter-Final-4', 'Pre-Quarter-7', 'Bracket-13'], self.groupWinners['E'][1])
		add_element(self.knockoutTree, ['Semi-Final-2', 'Quarter-Final-4', 'Pre-Quarter-7', 'Bracket-14'], self.groupWinners['F'][0])
		add_element(self.knockoutTree, ['Semi-Final-2', 'Quarter-Final-4', 'Pre-Quarter-8', 'Bracket-15'], self.groupWinners['G'][1])
		add_element(self.knockoutTree, ['Semi-Final-2', 'Quarter-Final-4', 'Pre-Quarter-8', 'Bracket-16'], self.groupWinners['H'][0])

		# print(self.knockoutTree)

	def predictKnockouts(self):

		stage = 4

		self.filledBracket = {0:[], 1:[], 2:[], 3:[]}

		def recursivePredict(tree, stage):
			# print(tree, '\n')
			stage -= 1
			if not isinstance(list(tree.values())[0], str):
				for item in tree:
					tree[item] = recursivePredict(tree[item], stage)	
			eG1 = poisson.rvs(self.expectedGoals[list(tree.values())[0]])
			eG2 = poisson.rvs(self.expectedGoals[list(tree.values())[1]])
			self.filledBracket[stage].append([list(tree.values())[0], eG1, list(tree.values())[1], eG2])
			if eG1 > eG2:
				return list(tree.values())[0]
			elif eG1 < eG2:
				return list(tree.values())[1]
			else:
				return np.random.choice(list(tree.values()))

		print(recursivePredict(self.knockoutTree, stage), self.filledBracket) 
							 

if __name__ == '__main__':
	W = WorldCupPredictor('/home/parth/Projects/UCSD/ProbabilityAndStatistics/Project/qualification_data')
	W.computeExpectedTeamGoals()
	W.predictGroupStages()
	W.createKnockoutTree()
	W.predictKnockouts()