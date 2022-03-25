# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 09:43:45 2022

@author: melan
"""


import psycopg2
import secrets_melanie
import time
from datetime import datetime

import pandas as pd
import numpy as np

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults, VARResultsWrapper
from statsmodels.tools.eval_measures import rmse, aic

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

import sklearn.metrics

import seaborn as sns
import matplotlib.pyplot as plt


import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
# render plot in default browser
pio.renderers.default = 'browser'


from urllib.request import urlopen
import json





pd.set_option('display.max_columns',None)

#%%

secrets = secrets_melanie.secrets()

conn = psycopg2.connect(host=secrets['db_url'],
        port=secrets['port'],
        dbname=secrets['db_name'],
        user=secrets['username'],
        password=secrets['password'],
        connect_timeout=10)

cur = conn.cursor()
#%%
# County hpi
query14 = """
select 
county_fips,
year,
annual_change_pct

from fhfa_house_price_index
"""

hpi_data = pd.read_sql(query14, con=conn)

# Redfin 
query15 = """
select 
county_fips,
period_end, 
property_type, 
median_sale_price,
median_sale_price_mom,
median_ppsf,
median_ppsf_mom

from redfin_county_full

where property_type ='All Residential'

"""

redfin_df = pd.read_sql(query15, con=conn)


conn.close()

#%% clean up redfin data
# Create date columns for filtering.  We will baseline the year over year change using the December records.
redfin = redfin_df.copy(deep=True)
redfin['date'] = pd.to_datetime(redfin['period_end'])
redfin['year'] = redfin['date'].dt.year
redfin['month'] = redfin['date'].dt.month

#%% Create date df for sequential dates for all county fips
counties = list(redfin.county_fips.unique())
years = list(range(2012,2022))
months =list(range(1,13))
base = pd.DataFrame([[counties],[years],[months]]).T
base = base.explode(0)
base = base.explode(1)
base = base.explode(2).reset_index(drop=True)
base.columns=['county_fips','year','month']
base = base[~((base['year']==2012)&(base['month']==1))]

base = base.merge(redfin,how='left',on=['county_fips','year','month'])
base = base.sort_values(by=['county_fips','year','month']).reset_index(drop=True)








#%%
#count number of records for each county
# county_records = df.groupby(['county_fips'],as_index=False)['year'].count()
# # count number of records for each county that have no median sale price yoy 
# county_isna = df[df['median_sale_price'].isna()].groupby(['county_fips'],as_index=False)['year'].count()
# # merge county counts and calculate % missing
# county_records = county_records.merge(county_isna, how='left', on=['county_fips',])
# county_records.columns = ['county_fips','total_records','total_isna']
# county_records = county_records.fillna(0)
# county_records['pct_isna'] = (county_records['total_isna']/county_records['total_records'])*100

# # find the month that has the most records with the least amount of na
# month_records = county_records.groupby(['year','month'],as_index=False)['total_records'].sum()
# month_isna = county_records.groupby(['year','month'],as_index=False)['total_isna'].sum()
# month_records = month_records.merge(month_isna, how='left', on=['year','month'])
# month_records['pct_isna'] = (month_records['total_isna']/month_records['total_records'])*100
# # sort by highest % missing
# month_records =month_records.sort_values(by=['total_records','pct_isna'])



# df = df.merge(redfin,how='left',on=['year','month']).reset_index(drop=True)
# df = df.sort_values(by=['county_fips','year','month'])

#%%


# # Find remaining columns that contain null values 
# for col in redfin.columns:
#     print(col+':', redfin[col].isnull().sum(), 'null values')

# redfin.shape
# #15017,18    
# #%%
# # how many counties have missing median sale price yoy missing for 2012
# redfin[((redfin['year']==2012)&(redfin['median_sale_price_yoy'].isna()))].shape
# #213
# redfin = redfin[~((redfin['year']==2012)&(redfin['median_sale_price_yoy'].isna()))]
# redfin.shape
# #14804,18

# # how many counties have the highest na
# #count number of records for each county
# county_records = redfin.groupby(['county_fips'],as_index=False)['year'].count()
# # count number of records for each county that have no median sale price yoy 
# county_isna = redfin[redfin['median_sale_price_yoy'].isna()].groupby(['county_fips'],as_index=False)['year'].count()
# # merge county counts and calculate % missing
# county_records = county_records.merge(county_isna, how='left', on='county_fips')
# county_records.columns = ['county_fips','total_records','total_isna']
# county_records = county_records.fillna(0)
# county_records['pct_isna'] = (county_records['total_isna']/county_records['total_records'])*100
# # sort by highest % missing
# county_records =county_records.sort_values(by='pct_isna',ascending=False)

# # assuming we need to have 3 non-

#%%

# df = pd.read_excel('C:/Users/melan/repo/Capstone/CapstoneTeamJim/MS-scratchpads/HPI_AT_BDL_county (1).xlsx', header = 6)
# from fredapi import Fred

# key = '432422d56877fe538ebf5a04d2f23ce5'
# fred = Fred(api_key=key)
#%%
# hpi=hpi_data.copy(deep=True)

# hpi['year'] = hpi['year'].astype(int)
# # time series needs datetime for index
# hpi['date'] = pd.to_datetime(hpi['year'], format = "%Y")
# # pd.to_datetime(hpi['year'], yearfirst=True)
# hpi['county_fips'] = hpi['county_fips'].astype(str)
# hpi['annual_change_pct'] = hpi['annual_change_pct'].astype(float)
# # hpi['year'] = hpi['year'].apply(lambda x: x-1)
# #keep only the last 10 years
# hpi = hpi[hpi['year']>=2009]
# # HPI is missing for 2019 and 2020 for select counties so dropping those
# hpi['year'] = hpi['year'].astype(str)
# hpi_piv = hpi.pivot(index='county_fips',columns = 'year',values = 'annual_change_pct')

# hpi_missing = list(hpi_piv[hpi_piv['2020'].isna()].index)
# hpi_missing += list(hpi_piv[hpi_piv['2019'].isna()].index)
# hpi = hpi[~hpi['county_fips'].isin(hpi_missing)]
#%%
# Using redfin median sale price for time series
df = base.copy(deep=True)
df['day']=1
# time series needs datetime for index
df['date'] = pd.to_datetime(df[['year','month','day']])
df['year'] = df['year'].astype(str)
df = df[['county_fips','date','median_sale_price','year']]

# Determine how many counties have missing values in training df and how many have recent missing (ie from 2019 to 2020)

missing_vals = df[df['median_sale_price'].isna()].fillna(1).groupby(['county_fips'],as_index=False)['median_sale_price'].count()
missing_vals.columns=['county_fips','total_missing']

recent_vals = df[df['date'].dt.year >=2019]
recent_vals = recent_vals[recent_vals['median_sale_price'].isna()].fillna(1).groupby(['county_fips'],as_index=False)['median_sale_price'].count()
recent_vals.columns=['county_fips','recent_missing']
missing_vals = missing_vals.merge(recent_vals, how='left',on='county_fips').fillna(0)
missing_vals['missing_pct'] = missing_vals['total_missing']/119*100
missing_vals['recent_pct'] = missing_vals['recent_missing']/36*100

missing_vals['to_filter'] = 1
# missing_vals['to_filter'][((missing_vals['missing_pct']<30)|(missing_vals['recent_pct']<30))]=0
missing_vals['to_filter'][((missing_vals['missing_pct']<10)|(missing_vals['recent_pct']<10))]=0
missing_counties = missing_vals[missing_vals['to_filter']==1]['county_fips'].to_list()
len(missing_counties)
# dropping 348 counties with 30:30
# dropping 433 counties with 10:10

df = df[~df['county_fips'].isin(missing_counties)]
df['median_sale_price'] = df.groupby('county_fips')['median_sale_price'].transform(lambda v: v.ffill()).fillna(0)

# split training and testing data and pivot so counties are columns
df_time = df[df['year']!='2021']


df_time = df_time.pivot(index='date',columns='county_fips',values='median_sale_price')


# counties that have no values in training vectors to be removed from both training and testing df
# missing_counties=df_time.sum().reset_index()
# missing_counties = list(missing_counties[missing_counties[0]==0]['county_fips'])
# df_time = df_time.drop(columns=missing_counties)
# df2021 = df2021.drop(columns=missing_counties)


#%%
def adfuller_func(dataframe):
    res = []
    for col in list(dataframe.columns):
        r = adfuller(dataframe[col], autolag='AIC')
        res.append(round(r[1],3))
    return res


# check stationarity with ad fuller testing - no differencing
no_diff = adfuller_func(df_time)
adf = pd.DataFrame(data=[no_diff],columns=list(df_time.columns))
adf = adf.melt()
adf.columns=['county_fips','no_diff_p']
adf[adf['no_diff_p']<0.05].count()
# 376/1512 are stationary - with 30:30 dropped counties
# 308/1512 are stationary- with 10:10 dropped counties

# check stationarity with ad fuller testing - first differencing
df_time_diff = df_time.diff().dropna()
first_diff = adfuller_func(df_time_diff)
adf['first_diff_p'] = first_diff 
adf[adf['first_diff_p']<0.05].count()
# 1404/1512 are stationary - with 30:30 dropped counties
# 1319/1512 are stationary- with 10:10 dropped counties

df_time_2diff = df_time_diff.diff().dropna()
sec_diff = adfuller_func(df_time_2diff)
adf['sec_diff_p'] = sec_diff 
adf[adf['sec_diff_p']<0.05].count()
# 1512/1512 are stationary - with 30:30 dropped counties
# 1427/1512 are stationary- with 10:10 dropped counties

# df = hpi[hpi['year']!='2020']
# d2020 = hpi[hpi['year']=='2020']
# df = df.pivot(index='date',columns='county_fips',values='annual_change_pct')
# d2020 = d2020.pivot(index='date',columns='county_fips',values='annual_change_pct')
# # Drop the county where the entire county has no HPI
# df = df.dropna(axis=1,how='all')
#%%
# missing_cols = []
# for col in df.columns:
#     if df[col].isna().sum() >0:
#         missing_cols.append(col)
# # drop counties where any year is missing HPI
# df = df.drop(columns=missing_cols)
# d2020 = d2020.drop(columns=missing_cols)

p=2

num_forecasts = 12
var_res, forecasts = None, None
# df_time_2diff = df_time_2diff+0.000000001

# using second differencing
model = VAR(df_time_2diff,freq='MS')


var_res = model.fit(maxlags=p)
# var_res.summary()
lag_order = var_res.k_ar
forecasts = var_res.forecast(df_time_2diff.values,steps=num_forecasts)
dfindex = pd.date_range(start=df_time.index[-1],periods = num_forecasts+1, freq='MS')[1:]
lastvals_second_diff = df_time.diff()[-1:]
for_df = pd.DataFrame(data = forecasts,columns= df_time.columns, index = dfindex)
for_df = pd.concat([lastvals_second_diff, for_df],axis=0,ignore_index=False).cumsum()[1:]
lastvals_first_diff = df_time[-1:]
for_df = pd.concat([lastvals_first_diff, for_df],axis=0,ignore_index=False).cumsum()[1:]

# df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
#%%
# compare predictions against actuals
df2021 = df[df['year']=='2021']

pred = pd.melt(for_df.reset_index(),id_vars='index',value_name = 'Pred_median_sale_price')
pred.columns=['date','county_fips','Pred_median_sale_price']
pred = pred.merge(df2021, on=['county_fips','date'])



# Calculate Error
df2021 = df2021.pivot(index='date',columns='county_fips',values='median_sale_price')
rmse_calc = np.sqrt(np.mean((df2021 - for_df) ** 2, axis=0)).round(0)
rmse_calc = rmse_calc.reset_index()
rmse_calc.columns = ['county_fips','rmse']
county_mean = pred.groupby('county_fips')['median_sale_price'].mean()
rmse_calc = rmse_calc.merge(county_mean, on=['county_fips'])
rmse_calc.columns = ['county_fips','rmse','mean_med_sale_by_county']
rmse_calc['error_magnitude'] = rmse_calc['rmse']/rmse_calc['mean_med_sale_by_county']*100

pred = pred.merge(rmse_calc,how='left',on='county_fips')
pred=pred.sort_values(by='error_magnitude')
#%%


#%%
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
    
fig = px.choropleth(rmse_calc, geojson=counties, locations='county_fips', color='error_magnitude',
                           color_continuous_scale="Viridis",
                            range_color=(0, 100),
                           scope="usa",
                           labels={'error_magnitude':'RMSE % of MEDIAN SALE PRICE'}
                          )
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

#%%


#####  How many counties are predicting median sale price <0 as a function of max_lags value?  
pred[pred['Pred_median_sale_price']<0]['county_fips'].nunique()
# p=3
#negative predictions counties (count=81)

# p=2
#negative predictions counties (count=18) with 30:30 dropped counties
#negative predictions counties (count=2) with 10:10 dropped counties

# p=1
#negative predictions counties (count=24)

# p=4
#negative predictions counties (count=54)

####  Setting max_lags to 2, how many counties have error magnitude >50% of median sale price?
pred[pred['error_magnitude']>50]['county_fips'].unique()
# counties with high error (>50% of median sale price) (count=109)  with 30:30 dropped counties
# '47007', '05149', '21181', '40079', '39157', '40063', '05061',
#    '51115', '28163', '48255', '27073', '48175', '13279', '19009',
#    '26051', '21161', '37005', '32015', '05009', '08021', '55085',
#    '40105', '28129', '05105', '51133', '08057', '05043', '45089',
#    '45047', '05017', '40095', '13091', '55125', '45041', '40015',
#    '13167', '40023', '16003', '19117', '48007', '39131', '13079',
#    '05049', '48391', '31095', '26035', '21077', '05037', '45071',
#    '37035', '22091', '49009', '19129', '40005', '12079', '05025',
#    '40077', '51181', '18115', '05121', '40069', '27071', '48429',
#    '47067', '55041', '27081', '13191', '40107', '39163', '29179',
#    '21135', '39145', '32027', '48163', '48031', '05079', '37009',
#    '05147', '05001', '05067', '21129', '35047', '47175', '48083',
#    '13095', '13301', '48333', '40035', '51103', '29061', '48041',
#    '13235', '13165', '28149', '48385', '05039', '51119', '40049',
#    '27031', '05035', '51131', '40127', '48237', '13127', '17013',
#    '48353', '48171', '48051', '18041'
# counties with high error (>50% of median sale price) (count=40)  with 10:10 dropped counties
# '19117', '45041', '21095', '21175', '40079', '40105', '05043',
#    '05011', '05097', '48429', '05061', '55041', '40107', '21181',
#    '40095', '13109', '47181', '48333', '27071', '45071', '13165',
#    '13127', '31095', '19009', '05121', '21077', '12079', '27081',
#    '19129', '48237', '40005', '05079', '13191', '13095', '40035',
#    '13235', '29061', '51119', '17013', '18041'


####  CHECK OUTLIERS    #####
# 18041
d18041 = base[base.county_fips=='18041']
m18041 = missing_vals[missing_vals['county_fips']=='18041']
#missing% = 5%; recent% = 14%

# 48051
d48051 = base[base.county_fips=='48051']
m48051 = missing_vals[missing_vals['county_fips']=='48051']
#missing% = 44%; recent% = 17%

# 48171
d48171 = base[base.county_fips=='48171']
m48171 = missing_vals[missing_vals['county_fips']=='48171']
#missing% = 33%; recent% = 19%

# 48353
d48353 = base[base.county_fips=='48353']
m48353 = missing_vals[missing_vals['county_fips']=='48353']
#missing% = 24%; recent% = 11%


# 25025
d25025 = base[base.county_fips=='25025']
m25025 = missing_vals[missing_vals['county_fips']=='25025']
#missing% = 0%; recent% = 0%

























