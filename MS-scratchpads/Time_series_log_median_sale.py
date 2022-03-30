# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 09:43:45 2022

@author: melan
"""


import psycopg2
import secrets_melanie


import pandas as pd
import numpy as np


from statsmodels.tsa.stattools import adfuller


from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults, VARResultsWrapper

from statsmodels.tsa.stattools import grangercausalitytests


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

# Create date df for sequential dates for all county fips
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
# Using redfin median sale price for time series
df = base.copy(deep=True)
df['day']=1
# time series needs datetime for index
df['date'] = pd.to_datetime(df[['year','month','day']])
df['year'] = df['year'].astype(str)
df = df[['county_fips','date','median_sale_price','year']]

# Determine how many counties have missing values in training df and how many have recent missing

missing_vals = df[df['median_sale_price'].isna()].fillna(1).groupby(['county_fips'],as_index=False)['median_sale_price'].count()
missing_vals.columns=['county_fips','total_missing']

# cutting off data - starting now at 2016 so checking where missing counties have missing values
recent_vals = df[df['date'].dt.year >=2016]
recent_vals = recent_vals[recent_vals['median_sale_price'].isna()].fillna(1).groupby(['county_fips'],as_index=False)['median_sale_price'].count()
recent_vals.columns=['county_fips','recent_missing']
missing_vals = missing_vals.merge(recent_vals, how='left',on='county_fips').fillna(0)
missing_vals['missing_pct'] = missing_vals['total_missing']/119*100
missing_vals['recent_pct'] = missing_vals['recent_missing']/72*100 #(6 years x 12 months)

missing_vals['to_filter'] = 1
missing_vals['to_filter'][((missing_vals['recent_pct']<10))]=0
missing_counties = missing_vals[missing_vals['to_filter']==1]['county_fips'].to_list()
len(missing_counties)
# dropping 467 counties with recent vals <10% in last 6 years.

# remove counties with sparse data and fill forward na for median sale price and backfill any values that didn't start in Jan 2016
df = df[~df['county_fips'].isin(missing_counties)]
df['median_sale_price'] = df.groupby('county_fips')['median_sale_price'].transform(lambda v: v.ffill()).fillna(method="bfill")
# Predicting on log of median sale price
df['log_median_sale_price'] = np.log(df['median_sale_price'])
# split training and testing data and pivot so counties are columns


df_time = df[((df['year']>='2016')&(df['year']<'2021'))]
df_time_log = df_time.pivot(index='date',columns='county_fips',values='log_median_sale_price')


#%%
def adfuller_func(dataframe):
    res = []
    for col in list(dataframe.columns):
        r = adfuller(dataframe[col], autolag='AIC')
        res.append(round(r[1],3))
    return res


# check stationarity with ad fuller testing - no differencing
# no_diff = adfuller_func(df_time_log)
# adf = pd.DataFrame(data=[no_diff],columns=list(df_time_log.columns))
# adf = adf.melt()
# adf.columns=['county_fips','no_diff_p']
# adf[adf['no_diff_p']<0.05].count()
# 451/1393 are stationary

# check stationarity with ad fuller testing - first differencing
df_time_diff_log = df_time_log.diff().dropna()
# first_diff = adfuller_func(df_time_diff_log)
# adf['first_diff_p'] = first_diff 
# adf[adf['first_diff_p']<0.05].count()
# 1348/1393 are stationary

df_time_2diff_log = df_time_diff_log.diff().dropna()
# sec_diff = adfuller_func(df_time_2diff_log)
# adf['sec_diff_p'] = sec_diff 
# adf[adf['sec_diff_p']<0.05].count()
# 1390/1393 are stationary



#%%
p=12

num_forecasts = 12
var_res, forecasts = None, None
# df_time_2diff = df_time_2diff+0.000000001

# using second differencing
model = VAR(df_time_2diff_log,freq='MS')


var_res = model.fit(maxlags=p)
# var_res.summary()
lag_order = var_res.k_ar
forecasts = var_res.forecast(df_time_2diff_log.values,steps=num_forecasts)
dfindex = pd.date_range(start=df_time_log.index[-1],periods = num_forecasts+1, freq='MS')[1:]
lastvals_second_diff = df_time_log.diff()[-1:]
for_df = pd.DataFrame(data = forecasts,columns= df_time_log.columns, index = dfindex)
for_df = pd.concat([lastvals_second_diff, for_df],axis=0,ignore_index=False).cumsum()[1:]
lastvals_first_diff = df_time_log[-1:]
for_df = pd.concat([lastvals_first_diff, for_df],axis=0,ignore_index=False).cumsum()[1:]


#%%
###  
df2021 = df[df['year']=='2021']


pred = pd.melt(for_df.reset_index(),id_vars='index',value_name = 'Pred_log_median_sale_price')
pred.columns=['date','county_fips','Pred_log_median_sale_price']
pred = pred.merge(df2021, on=['county_fips','date'])
# calculate the log_errors
pred['log_pred_errors'] = pred['log_median_sale_price']-pred['Pred_log_median_sale_price']
hist = px.histogram(pred['log_pred_errors'])
hist.show()


###  Plot residual errors  (still in log transform)
# https://www.qualtrics.com/support/stats-iq/analyses/regression-guides/interpreting-residual-plots-improve-regression/
## Keep - shows that errors 
# (1) they’re pretty symmetrically distributed, tending to cluster towards the middle of the plot.
# (2) they’re clustered around the lower single digits of the y-axis (e.g., 0.5 or 1.5, not 30 or 150).
# (3) in general, there aren’t any clear patterns.
pred_errors = px.scatter(pred,x='log_median_sale_price',y='log_pred_errors')
pred_errors.show()

###   Transform log predictions to original units

# https://stats.stackexchange.com/questions/55692/back-transformation-of-an-mlr-model

# calculate the mean transformed error by county
pred['pred_errors_trans'] = np.exp(pred['log_pred_errors'])
mean_transformed_error = pred.groupby('county_fips')['pred_errors_trans'].mean().reset_index(name='Mean_pred_error_trans')

# transform predicted median sale price back to dollars
pred['pred_med_sale_trans_biased'] = np.exp(pred['Pred_log_median_sale_price'])
pred = pred.merge(mean_transformed_error,on='county_fips')
pred['pred_med_sale_trans']  = pred['pred_med_sale_trans_biased']*pred['Mean_pred_error_trans']


# Prediction by county for 2021
county_mean_pred2021 = pred.groupby('county_fips')['pred_med_sale_trans'].mean().reset_index(name='Predicted_mean_2021')
county_mean = pred.groupby('county_fips')['median_sale_price'].mean().reset_index(name='County_mean')
Predictions_by_county = county_mean_pred2021.merge(county_mean,on='county_fips')
Predictions_by_county['error'] = Predictions_by_county['County_mean']-Predictions_by_county['Predicted_mean_2021']
Predictions_by_county['error_pct'] = np.absolute(Predictions_by_county['error'])/Predictions_by_county['County_mean']*100



#%% Granger Causality checks
# from tqdm import tqdm
# df_gran_test = df.pivot(index='date',columns='county_fips',values='median_sale_price')
# counties = list(df_gran_test.columns)
# test = 'ssr_chi2test'
# granger_df = pd.DataFrame(data=None, columns = counties, index = counties)

# for i in tqdm(counties):
#     for j in counties:
#         if i!=j:
#             test_result = grangercausalitytests(df_gran_test[[i, j]], maxlag=p, verbose=False)
#             p_values = [round(test_result[i+1][0][test][1],4) for i in range(p)]
#             min_p_value = np.min(p_values)
#             granger_df.loc[i,j] = min_p_value
            
# granger_df.astype(float)

#Takes about 4.5 hours to run.  It can be seen that there are several series that have causality to other time series.

#%%
# from statsmodels.tsa.vector_ar.vecm import coint_johansen


# def cointegration_test(df, alpha=0.05): 
#     """Perform Johanson's Cointegration Test and Report Summary"""
#     out = coint_johansen(df,-1,2)
#     d = {'0.90':0, '0.95':1, '0.99':2}
#     traces = out.lr1
#     cvts = out.cvt[:, d[str(1-alpha)]]
#     def adjust(val, length= 6): return str(val).ljust(length)

#     # Summary
#     print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
#     for col, trace, cvt in zip(df.columns, traces, cvts):
#         print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
# df_coint = df.pivot(index='date',columns='county_fips',values='median_sale_price')

# cointegration_test(df_coint.iloc[:,0:5])

#%%
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
    
fig = px.choropleth(Predictions_by_county, geojson=counties, locations='county_fips', color='error_pct',
                           color_continuous_scale="Viridis",
                            # range_color=(0, 50),
                           scope="usa",
                           labels={'error_pct':'% Prediction error'}
                          )
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

#%%


####  CHECK OUTLIERS    #####
# 29061
d29061 = base[base.county_fips=='29061']



























