import numpy as np
import pandas as pd
from scrape_data import *
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
plt.style.use('ggplot')
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
from collections import Counter

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
		# print(self.expectedGoals)

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

		self.filledBracket = {}

		def recursivePredict(tree, stage):
			stage -= 1
			if not isinstance(list(tree.values())[0], str):
				for item in tree:
					tree[item] = recursivePredict(tree[item], stage)	
			eG1 = poisson.rvs(self.expectedGoals[list(tree.values())[0]])
			eG2 = poisson.rvs(self.expectedGoals[list(tree.values())[1]])
			self.filledBracket[list(tree.keys())[0]] = [list(tree.values())[0]]
			self.filledBracket[list(tree.keys())[1]] = [list(tree.values())[1]]
			if eG1 > eG2:
				return list(tree.values())[0]
			elif eG1 < eG2:
				return list(tree.values())[1]
			else:
				return np.random.choice(list(tree.values()))

		self.filledBracket['Winner'] = [recursivePredict(self.knockoutTree, stage)]

	def runSimulation(self, n = 10000):

		for i in range(n):
			self.predictGroupStages()
			self.createKnockoutTree()
			self.predictKnockouts()

			try:
				for item in self.stageCounts:
					self.stageCounts[item] += self.filledBracket[item]
			except AttributeError:
				self.stageCounts = self.filledBracket

		for item in self.stageCounts:
			self.stageCounts[item] = {k:v/len(self.stageCounts[item]) for k, v in Counter(self.stageCounts[item]).items()}
	
		print(dict(sorted(self.stageCounts['Winner'].items(), key=lambda item: item[1])))

	def plotBracket(self):
		groups = ['A', 'C', 'E', 'G']
		for i in range(1, 8, 2):
			plt.clf()
			D = self.stageCounts['Bracket-{}'.format(i)]
			plt.bar(range(len(D)), list(D.values()), align='center', color = list(plt.rcParams['axes.prop_cycle'])[0]['color'])
			plt.ylim([0, 1])
			plt.xticks(range(len(D)), list(D.keys()))
			plt.title('Group {} Winner Probability'.format(groups[i//2]))
			plt.savefig('images/Group {} Winners.png'.format(groups[i//2]))

		groups = ['A', 'C', 'E', 'G']
		for i in range(9, 16, 2):
			plt.clf()
			D = self.stageCounts['Bracket-{}'.format(i)]
			plt.bar(range(len(D)), list(D.values()), align='center', color = list(plt.rcParams['axes.prop_cycle'])[0]['color'])
			plt.ylim([0, 1])
			plt.xticks(range(len(D)), list(D.keys()))
			plt.title('Group {} Runners-Up Probability'.format(groups[(i-9)//2]))
			plt.savefig('images/Group {} Runners-Up.png'.format(groups[(i-9)//2]))

		groups = ['B', 'D', 'F', 'H']
		for i in range(10, 17, 2):
			plt.clf()
			D = self.stageCounts['Bracket-{}'.format(i)]
			plt.bar(range(len(D)), list(D.values()), align='center', color = list(plt.rcParams['axes.prop_cycle'])[0]['color'])
			plt.ylim([0, 1])
			plt.xticks(range(len(D)), list(D.keys()))
			plt.title('Group {} Winner Probability'.format(groups[(i-9)//2]))
			plt.savefig('images/Group {} Winners.png'.format(groups[(i-9)//2]))

		groups = ['B', 'D', 'F', 'H']
		for i in range(2, 9, 2):
			plt.clf()
			D = self.stageCounts['Bracket-{}'.format(i)]
			plt.bar(range(len(D)), list(D.values()), align='center', color = list(plt.rcParams['axes.prop_cycle'])[0]['color'])
			plt.ylim([0, 1])
			plt.xticks(range(len(D)), list(D.keys()))
			plt.title('Group {} Runners-Up Probability'.format(groups[(i-9)//2]))
			plt.savefig('images/Group {} Runners-Up.png'.format(groups[(i-9)//2]))

		for i in range(1, 9):
			plt.clf()
			D = self.stageCounts['Pre-Quarter-{}'.format(i)]
			plt.bar(range(len(D)), list(D.values()), align='center', 
				color = list(plt.rcParams['axes.prop_cycle'])[1]['color'])
			plt.ylim([0, 1])
			plt.xticks(range(len(D)), list(D.keys()), rotation=45, ha='right')
			plt.title('Pre-Quarter-{} Winner Probability'.format(i))
			plt.tight_layout()
			plt.savefig('images/Pre-Quarter-{} Winners.png'.format(i))

		for i in range(1, 5):
			plt.clf()
			D = self.stageCounts['Quarter-Final-{}'.format(i)]
			plt.bar(range(len(D)), list(D.values()), align='center', color = list(plt.rcParams['axes.prop_cycle'])[2]['color'])
			plt.ylim([0, 1])
			plt.xticks(range(len(D)), list(D.keys()), rotation=45, ha='right')
			plt.title('Quarter-Final-{} Winner Probability'.format(i))
			plt.tight_layout()
			plt.savefig('images/Quarter-Final-{} Winners.png'.format(i))

		for i in range(1, 3):
			plt.clf()
			D = self.stageCounts['Semi-Final-{}'.format(i)]
			plt.bar(range(len(D)), list(D.values()), align='center', color = list(plt.rcParams['axes.prop_cycle'])[3]['color'])
			plt.ylim([0, 1])
			plt.xticks(range(len(D)), list(D.keys()), rotation=45, ha='right')
			plt.title('Semi-Final-{} Winner Probability'.format(i))
			plt.tight_layout()
			plt.savefig('images/Semi-Final-{} Winners.png'.format(i))

		plt.clf()
		D = self.stageCounts['Winner']
		plt.bar(range(len(D)), list(D.values()), align='center', color = list(plt.rcParams['axes.prop_cycle'])[4]['color'])
		plt.ylim([0, 1])
		plt.xticks(range(len(D)), list(D.keys()), rotation=45, ha='right')
		plt.title('Winner Probability')
		plt.tight_layout()
		plt.savefig('images/Winners.png')

if __name__ == '__main__':
	W = WorldCupPredictor('/home/parth/Projects/UCSD/ProbabilityAndStatistics/Project/qualification_data')
	W.computeExpectedTeamGoals()
	W.runSimulation()
	# W.plotBracket()