# For Club Model
# Player in Messi, Ronaldo, Mbappe, Griezman, Kane, Perisic, Neymar, Morata
# Get 20-21 Club, 21-22 Club
# For 2021Club, 2122Club, get all competition fixtures
# For fixture, if fixutre in League/CL, go to match report
# Get team : xg/10, npxg/10, xag/10, sca/10, cmp/10, prog/10. Get opossing team : save%, tackle/10, cmp/10
# Get player xg


import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re

players = {'Lionel-Messi': 'd70ce98e',
           'Cristiano-Ronaldo': 'dea698d9',
           'Kylian-Mbappe': '42fd9c7f',
           'Antoine-Griezmann': 'df69b544',
           'Harry-Kane': '21a66f6a',
           'Neymar': '69384e5d'}

player_to_team_prev = {'Lionel-Messi': '206d90db'}
player_to_team_curr = {'Lionel-Messi': 'e2d8892c'}


def get_team_id(player, year):
    if year in ['2018-2019', '2019-2020', '2020-2021']:
        return player_to_team_prev[player]
    else:
        return player_to_team_curr[player]


def get_player_url(player):
    return "https://fbref.com/en/players/" + players[player] + "/" + player


player_to_club = {'Lionel-Messi': 'e2d8892c',
                  'Cristiano-Ronaldo': '19538871',
                  'Kylian-Mbappe': 'e2d8892c',
                  'Neymar': 'e2d8892c',
                  'Antoine-Griezmann': 'db3b9613',
                  'Harry Kane': '361ca564'}


def get_club_url(player, year='2021-2022'):
    return 'https://fbref.com/en/squads/' + player_to_club[player] + '/' + year + '/all_comps/'


def get_player_match_log_url(player, year='2021-2022'):
    return 'https://fbref.com/en/players/' + players[player] + \
           '/matchlogs/' + year + '/summary/' + player + '-Match_logs'


def filter_matches(player, year='2021-2022', comp='club'):
    res = requests.get(get_player_match_log_url(player, year))
    ## The next two lines get around the issue with comments breaking the parsing.
    comm = re.compile("<!--|-->")
    soup = BeautifulSoup(comm.sub("", res.text), 'html.parser')
    all_tables = soup.findAll("tbody")
    match_logs = all_tables[0]
    rows = match_logs.findAll('tr')
    ret_rows = []
    if comp == 'club':
        comps = ['Premier League', 'Ligue 1', 'La Liga', 'Champions Lg', 'Conf Lg']
    else:
        comps = ['UEFA Euro', 'World Cup', 'Copa América']
    for r in rows:
        if r.find('td', {'data-stat': 'comp'}).getText() in comps:
            ret_rows.append(r)
    return ret_rows


def get_match_url(row):
    mr = row.find('td', {'data-stat': 'match_report'})
    end = mr.findAll('a', href=True)[0]['href']
    return 'https://fbref.com' + end


# For Club Model
# Player in Messi, Ronaldo, Mbappe, Griezman, Kane, Perisic, Neymar, Morata
# Get 20-21 Club, 21-22 Club
# For 2021Club, 2122Club, get all competition fixtures
# For fixture, if fixutre in League/CL, go to match report
# Get team : xg/10, npxg/10, xag/10, sca/10, cmp/10, prog/10. Get opossing team : save%, tackle/10, cmp/10
# Get player xg

def get_stat(row, stat):
    cells = row.findAll('td')
    for cell in cells:
        if cell['data-stat'] == stat:
            return 0.0 if cell.getText() == '' else float(cell.getText())
    return 0.0


def get_match_stats(url, player, year='2022-2023'):
    teamid = get_team_id(player, year)
    res = requests.get(url)
    ## The next two lines get around the issue with comments breaking the parsing.
    comm = re.compile("<!--|-->")
    soup = BeautifulSoup(comm.sub("", res.text), 'html.parser')
    # print(url)
    div = soup.find('div', {'id': 'div_stats_' + teamid + '_summary'})
    table = div.find('table', {'id': 'stats_' + teamid + '_summary'})
    rows = table.tbody.findAll('tr')
    for r in rows:
        if r.th['data-append-csv'] == players[player]:
            break
    stats = ['xg', 'npxg', 'xg_assist', 'sca', 'passes_pct', 'progressive_passes']
    player_stats = [get_stat(r, stat) for stat in stats]
    team_row = table.tfoot.tr
    team_stats = [get_stat(team_row, stat) for stat in stats]
    for i in range(len(stats)):
        if stats[i] == 'passes_pct':
            continue
        team_stats[i] -= player_stats[i]
        team_stats[i] /= 10
    return player_stats[0], team_stats + get_opp_team_stats(res, player, year)


def get_opp_team_stats(res, player, year='2022-2023'):
    teamid = get_opp_team_id(res, player, year)
    comm = re.compile("<!--|-->")
    soup = BeautifulSoup(comm.sub("", res.text), 'html.parser')
    # try:
    #     passdiv = soup.find('div', {'id': 'div_stats_' + teamid + '_summary'})
    #     passtable = passdiv.find('table', {'id': 'stats_' + teamid + '_summary'})
    #     team_row = passtable.tfoot.tr
    #     pass_stat = get_stat(team_row, 'passes_pct')
    # except Exception as e:
    #     pass_stat = np.nan

    try:
        keeperdiv = soup.find('div', {'id': 'div_keeper_stats_' + teamid})
        keepertable = keeperdiv.find('table', {'id': 'keeper_stats_' + teamid})
        row = keepertable.tbody.findAll('tr')[0]
        save_stats = get_stat(row, 'gk_save_pct')
    except Exception as e:
        save_stats = np.nan

    try:
        defdiv = soup.find('div', {'id': 'div_stats_' + teamid + '_defense'})
        deftable = defdiv.find('table', {'id': 'stats_' + teamid + '_defense'})
        stats = ['tackles', 'interceptions']
        team_row = deftable.tfoot.tr
        team_stats = [get_stat(team_row, stat) / 11 for stat in stats]
    except Exception as e:
        team_stats = [np.nan, np.nan]
    return [save_stats] + team_stats
    # return [save_stats] + [pass_stat] + team_stats


def get_opp_team_id(res, player, year='2022-2023'):
    teamid = get_team_id(player, year)
    comm = re.compile("<!--|-->")
    soup = BeautifulSoup(comm.sub("", res.text), 'html.parser')
    divs = soup.findAll('div')
    ids = [d.get('id', '') for d in divs]
    ids = list(filter(lambda x: re.match('^div_stats_(?P<id>\w+)_summary$', x), ids))
    if ids[0].split('_')[2] == teamid:
        return ids[1].split('_')[2]
    return ids[0].split('_')[2]


def create_club_data(player):
    matches = filter_matches(player) + filter_matches(player, year='2022-2023')
    urls = [get_match_url(r) for r in matches]
    stats = [get_match_stats(u, player, year) for u in urls]
    data = np.array(stats)
    y = data[:][0]
    x = data[:][1]
    x = np.array([np.array(a) for a in x])
    stats = ['xg', 'npxg', 'xg_assist', 'sca', 'passes_pct', 'progressive_passes']
    stats += ['saves', 'tackles', 'interceptions']
    df = pd.DataFrame(x, columns=stats)
    df['y'] = pd.Series(y)
    return x, y


def create_club_data_(player, year='2018-2019'):
    matches = filter_matches(player, year)
    urls = [get_match_url(r) for r in matches]
    stats = [get_match_stats(u, player, year) for u in urls]
    data = np.array(stats)
    y = data[:, 0]
    x = data[:, 1]
    x = np.array([np.array(a) for a in x])
    stats = ['xg', 'npxg', 'xg_assist', 'sca', 'passes_pct', 'progressive_passes']
    stats += ['saves', 'tackles', 'interceptions']
    df = pd.DataFrame(x, columns=stats)
    df['y'] = pd.Series(y)
    return df


def create_int_data_(player, year='2018-2019'):
    matches = filter_matches(player, year, comp='int')
    urls = [get_match_url(r) for r in matches]
    stats = [get_match_stats(u, player, year) for u in urls]
    data = np.array(stats)
    y = data[:, 0]
    x = data[:, 1]
    x = np.array([np.array(a) for a in x])
    stats = ['xg', 'npxg', 'xg_assist', 'sca', 'passes_pct', 'progressive_passes']
    stats += ['saves', 'tackles', 'interceptions']
    df = pd.DataFrame(x, columns=stats)
    df['y'] = pd.Series(y)
    return df


def merge_club_data(player, existing_data='Messi_Club_2021-2022-2023.csv'):
    stats = ['xg', 'npxg', 'xg_assist', 'sca', 'passes_pct', 'progressive_passes']
    stats += ['saves', 'tackles', 'interceptions']
    if existing_data:
        years = ['2018-2019', '2019-2020', '2020-2021']
    else:
        years = ['2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023']
    data = pd.concat([create_club_data_(player, year) for year in years], ignore_index=True)
    if existing_data:
        data_csv = pd.read_csv(existing_data)[stats + ['y']]
        data = pd.concat([data, data_csv], ignore_index=True)
    return data


def merge_int_data(player):
    global player_to_team_prev, player_to_team_curr
    player_to_team_prev = {'Lionel-Messi': 'f9fddd6e'}
    player_to_team_curr = {'Lionel-Messi': 'f9fddd6e'}
    stats = ['xg', 'npxg', 'xg_assist', 'sca', 'passes_pct', 'progressive_passes']
    stats += ['saves', 'tackles', 'interceptions']
    years = ['2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023']
    data = pd.concat([create_int_data_(player, year) for year in years], ignore_index=True)
    return data


# For international model
# Player in Messi, Ronaldo, Mbappe, Griezman, Kane, Perisic, Neymar, Morata
# Get 20-21 Club, 21-22 Club
# For 2021Club, 2122Club, get all competition fixtures
# For fixture, if fixutre in WC/Euro/CA, go to match report
# Get team : (xg, npxg, xag, sca, cmp, prog)/10. Get opossing team : save%, (tackle, cmp)/10
# stats = ['xg', 'npxg', 'xg_assist', 'sca', 'passes_pct', 'progressive_passes']
# Get player xg

# For predictions
# Player in Messi, Ronaldo, Mbappe, Griezman, Kane, Perisic, Neymar, Morata
# Get avg xg, npxg, xab, sca, cmp, prog over teammates
# Get avg opposing team save%, tackles, cmp
# Hand Collected Data
# Messi Opp Teams
# ['save%', 'pass_pct', 'tackles', 'interceptions']
# Poland : [60.0, 79.1, 15, 10.2]
# Mexico : [50.0, 78.8,16.7, 8.3]
# Saudi Arabia : [77.3, 73.7, 15.7, 10.3]
# Australia : [66.7, 75.9, 14.5, 10]
# Netherlands: [77.4, 86.2, 18.2, 10.4]
# Croatia: [80, 86.5, 18.6, 8]
# France: [83.3, 96.2, 19.6, 9.8]


# Predict

if __name__ == '__main__':
    player = 'Lionel-Messi'
    # ALready ran for club data
    # club_data = merge_club_data(player)
    player_to_team_prev = {'Lionel-Messi': 'f9fddd6e'}
    player_to_team_curr = {'Lionel-Messi': 'f9fddd6e'}
