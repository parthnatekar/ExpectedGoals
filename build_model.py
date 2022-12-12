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
from sklearn.impute import SimpleImputer
import matplotlib.mlab as mlab


class DataGenerator:

    def __init__(self, dataframe1, dataframe2=None):
        self.data = dataframe1
        self.data = self.data.add_prefix('club_')

        if dataframe2 is not None:
            self.data2 = dataframe2
            self.data2 = self.data2.add_prefix('international_')

        self.merged = pd.merge(self.data, self.data2, how='inner', left_on='club_player',
                               right_on='international_player')

    def getStatistics(self, column_name=''):
        column = self.data[column_name].values
        mean = np.mean(column)
        variance = np.var(column)
        return mean, variance

    def plotDistribution(self, column_name=''):
        column = self.merged[column_name].values
        plt.hist(column, bins=30)
        plt.show()

    def fitExpon(self, column_name=''):
        column = self.merged[column_name].values
        bins = np.linspace(0, column.max(), 100)
        entries, bin_edges = np.histogram(column, bins=bins, density=True)
        P = ss.expon.fit(entries)
        print(P)
        bin_middles = [bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2 for i in range(len(bin_edges) - 1)]
        plt.hist(column, bins=bins, density=True)
        rP = ss.expon.pdf(bin_middles, *P)
        plt.plot(bin_middles, rP)
        plt.show()

    def fitGaussian(self, column_name=''):
        column = self.merged[self.merged['club_minutes'] > 90][column_name].values
        column = column[(column > np.percentile(column, 5)) & (column < np.percentile(column, 95))]
        (mu, sigma) = ss.norm.fit(column)
        n, bins, patches = plt.hist(column, 60, density=True, facecolor='green', alpha=0.75)
        y = ss.norm.pdf(bins, mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=2)
        plt.show()

    def fitPoisson(self, column_name=''):
        # Can't fit poisson, discrete?
        def fit_function(k, lamb):
            '''poisson function, parameter lamb is the fit parameter'''
            return poisson.pmf(k, lamb)

        column = self.merged[self.merged['club_minutes'] > 90][column_name].values
        entries, bins, patches = plt.hist(column, np.arange(0, int(np.max(column)), 1), density=True, facecolor='green',
                                          alpha=0.75)
        bin_middles = np.arange(0, np.max(column) - 1) + 0.5
        print(bins)
        parameters, cov_matrix = curve_fit(fit_function, bins[:-1], entries)
        print(parameters)
        plt.plot(
            bins,
            fit_function(bins, *parameters), linestyle='-',
            label='Fit result', color='orange'
        )
        # plt.plot(bin_middles, entries)
        plt.show()

    def createTeamVariable(self, column_name):
        group_data = {}

        # Group data for each club team into a dictionary
        for team in self.merged.groupby(by=['club_team']).groups.keys():
            group_data[team] = self.merged.groupby(by=['club_team']).get_group(team)

        group_statistic = []
        cardinality = []

        # For each player in the data, creates a column of mean of club variable values of the club teammates of each player,
        # For example, for Cristiano Ronaldo, it creates a column of mean([Marcus Rashford MUN value, Antony MUN value, ...])

        for i in range(len(self.merged)):
            # Get the player team
            player_team = self.merged.iloc[i]['club_team']

            # Get only the data for that team, then remove that player from the data
            team_data_without_player = group_data[player_team][
                group_data[player_team]['club_player'] != self.merged.iloc[i]['club_player']]

            # Store the number of team members found
            cardinality.append(len(
                group_data[player_team][group_data[player_team]['club_player'] != self.merged.iloc[i]['club_player']]))

            # Append the mean of the team members variable value (given by column name argument)
            group_statistic.append(np.nanmean(team_data_without_player['club_' + column_name].values))

        # Add the new column to the dataframe
        self.merged['rest_of_club_team_{}'.format('club_' + column_name)] = pd.Series(group_statistic)
        self.merged['rest_of_club_team_{}'.format('cardinality')] = pd.Series(cardinality)

    ## The next code does the same, but replaces the club team of the player by the international team,
    ## i.e. it creates a column of mean of club variable values of the international teammates of each player,
    ## For example, for Cristiano Ronaldo, it creates a column of mean([Joao Felix ATM value, Bernardo Silva MCI, ...])

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
        return pd.merge(self.data, self.data2, left_on='club_player', right_on='international_player')


class BayesianModel:
    def __init__(self, L):
        self.L = L
        self.col_names = ['xg_per90', 'npxg_per90', 'shots_on_target_per90',
                     'xg_assist_per90', 'passes_completed', 'passes_into_final_third', 'passes_into_penalty_area',
                     'crosses_into_penalty_area', 'through_balls',
                     'touches_att_3rd', 'errors']
        for col in self.col_names:
            self.L.createTeamVariable(col)
        self.club_variables_dict = {'dependent': 'club_xg_per90',
                                  'independent': ['rest_of_club_team_club_'+x for x in self.col_names]}

    def buildClubModel(self):
        self.club_model = LinearModel(self.L.merged)
        self.club_model.buildLinearModel(self.club_variables_dict)
        return self.club_model



class LinearModel:

    def __init__(self, dataframe):
        self.data = dataframe

    def buildLinearModel(self, variables_dict, model_type='linear_regression'):
        y = self.data[variables_dict['dependent']].values
        # X = self.data.select_dtypes(include=np.number).apply(lambda x: x.fillna(x.mean()),axis=0).filter(regex='club')
        X = self.data[variables_dict['independent']].values
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imp.fit_transform(X)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=4)

        positions_list = ['FW']
        # print(self.data['club_position'].values in positions_list)

        # y = y[(self.data['rest_of_club_team_cardinality'].values > 8)]
        # X = X[(self.data['rest_of_club_team_cardinality'].values > 8)]

        self.reg = LinearRegression().fit(X_train, y_train)

        # coeff = pd.concat([pd.DataFrame(X.columns), pd.DataFrame(np.transpose(self.reg.coef_))], axis = 1)

        # coeff.to_csv('coeff.csv')

        print(self.reg.coef_, self.reg.score(X_test, y_test))

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

    # L.createTeamVariable('xg_assist_per90')
    # L.createTeamVariable('goals_per90')
    # L.createTeamVariable('xg_xg_assist_per90')
    # L.createTeamVariable('passes_pct')
    # L.createTeamVariable('passes_completed')
    # L.createTeamVariable('sca_per90')

    # L.merged['club_goals_minus_xg_per90'] = L.merged['club_goals_per90'] - L.merged['club_xg_per90']
    # L.merged['club_passes_per90'] = L.merged['club_passes']/L.merged['club_games']
    # L.merged['club_touches_att_3rd_per90'] = L.merged['club_touches_att_3rd']*90/L.merged['club_minutes']
    # L.merged.to_csv('Euro21_PL21_corrected.csv')
    # [team xg per 90, team xg against per 90, team x assist per 90, avg team possession, team through balls,
    # team touches att 3rd, team crosses into penalty area]
    col_names = ['xg_per90', 'npxg_per90', 'shots_on_target_per90',
                 'xg_assist_per90', 'passes_completed', 'passes_into_final_third', 'passes_into_penalty_area',
                 'crosses_into_penalty_area', 'through_balls',
                 'touches_att_3rd', 'errors']
    for col in col_names:
        L.createTeamVariable(col)

    fit_variables_dict = {'dependent': 'international_xg_per90',
                          'independent': ['club_npxg_per90', 'club_goals_assists_per90']}

    M = LinearModel(L.merged)
    M.buildLinearModel(fit_variables_dict)

# predict_variables_dict = {'dependent': 'international_goals_per90',
# 'independent': ['rest_of_international_team_club_xg_assist_per90',
# 			    'rest_of_international_team_club_sca_per90']}

# M.evaluateLinearModel(predict_variables_dict)

# L.plotDistribution('club_touches_att_3rd_per90')
# L.fitExpon('club_xg_per90')
# L.fitGaussian('club_passes_pct')
# L.fitPoisson('club_clearances')

# Calculate variance of goals scored by each team
# Use Xg as the expected value, variance calculated as above.
# Get a bound on the probability using Chebyshev.
# Calculate total lower bound probability of all group stage games.

# Fit different distributions to different statistics, get a bound on
# how far their values are from the actual means of the distribution.

# Hypothesis testing?
