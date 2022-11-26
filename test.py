from statsbombpy import sb
from scrape_data import *

df_outfield = get_outfield_data('https://fbref.com/en/comps/9/','/Premier-League-Stats')

df_outfield.to_csv('PL2021_Outfield.csv',index=False)

df_outfield_WC = get_outfield_data('https://fbref.com/en/comps/1/2018/','/2018-FIFA-World-Cup-Stats')


df_outfield_WC.to_csv('WC2018_Outfield.csv',index=False)

# For Goalkeepers, row names have changed, need to manually replace those before
# extracting

df_keeper = get_keeper_data('https://fbref.com/en/comps/9/','/Premier-League-Stats')

df_keeper.to_csv('PL2021_keepers.csv',index=False)