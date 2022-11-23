from statsbombpy import sb
from scrape_data import *

df_outfield = get_outfield_data('https://fbref.com/en/comps/9/','/Premier-League-Stats')

df_outfield.to_csv('PL2021_Outfield.csv',index=False)

# For Goalkeepers, row names have changed, need to manually replace those before
# extracting

df_keeper = get_keeper_data('https://fbref.com/en/comps/9/','/Premier-League-Stats')

df_keeper.to_csv('PL2021_keepers.csv',index=False)