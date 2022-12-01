from statsbombpy import sb
from scrape_data import *

df_outfield = get_outfield_data('https://fbref.com/en/comps/Big5/2020-2021/','/players/2020-2021-Big-5-European-Leagues-Stats')

df_outfield.to_csv('data/Big52020-21_Outfield.csv',index=False)

# df_outfield_WC = get_outfield_data('https://fbref.com/en/comps/676/','/European-Championship-Stats')


# df_outfield_WC.to_csv('Euro2021_Outfield.csv',index=False)

# For Goalkeepers, row names have changed, need to manually replace those before
# extracting

# df_keeper = get_keeper_data('https://fbref.com/en/comps/9/','/Premier-League-Stats')

# df_keeper.to_csv('PL2021_keepers.csv',index=False)