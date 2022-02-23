# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:07:48 2022

@author: melan
"""


import psycopg2
import secrets_melanie
import time
import myutils
import requests
import json

import pandas as pd
import numpy as np
import censusdata

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

import seaborn as sns


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
# Bring in datasets
# State Corporate Income tax - long version
query1 = """
select
state_fips,
year,
corp_income_tax_low,
corp_income_tax_high
 
from state_corp_income_tax_long
"""

corp = pd.read_sql(query1, con=conn)

# State Corporate Income tax - long version
query2 = """
select
state_fips,
year,
income_tax_low,
income_tax_high
from state_income_tax_long
"""

tax = pd.read_sql(query2, con=conn)

# County Debt ratio - long version
query3 = """
select * 
from county_debt_ratio_long
"""

debt = pd.read_sql(query3, con=conn)

# County vehicles - long version
query4 = """
select 
county_fips,
year,
vehicles_available

from acs1_vehicles_available
"""

vehicles = pd.read_sql(query4, con=conn)

# County travel_time- long version
query5 = """
select 
county_fips,
year,
travel_time_to_work

from acs1_travel_time_to_work
"""

travel = pd.read_sql(query5, con=conn)


# County population- long version
query6 = """
select 
county_fips,
year,
population

from acs1_population
"""

population = pd.read_sql(query6, con=conn)

# County household income- long version
query7 = """
select 
county_fips,
year,
household_income

from acs1_household_income
"""

income = pd.read_sql(query7, con=conn)

# County Median Home value- long version
query8 = """
select 
county_fips,
year,
home_value_median

from acs1_home_value_median
"""

home_value = pd.read_sql(query8, con=conn)

#%%
#join datasets

df = debt.merge(corp,how='left',on=['state_fips','year'])
df = df.merge(tax,how='left',on=['state_fips','year'])
df = df.merge(vehicles,how='left',on=['county_fips','year'])
df = df.merge(travel,how='left',on=['county_fips','year'])
df = df.merge(population,how='left',on=['county_fips','year'])
df = df.merge(income,how='left',on=['county_fips','year'])
df = df.merge(home_value,how='left',on=['county_fips','year'])

df = df.dropna()

#%%
#Bring in HPI target data
hpi_df = pd.read_excel('C:/Users/melan/HPI_AT_BDL_county.xlsx', encoding = 'utf-8',header=6, dtype=str)
hpi_df = hpi_df[['Year', 'FIPS code', 'Annual Change (%)']]
hpi_df.columns = ['year', 'county_fips', 'hpi_change_in_a_year']

#reduce year by 1 so that the hpi annual change value is for one year in the future
hpi_df['year'] = hpi_df['year'].astype(int)
hpi_df['county_fips'] = hpi_df['county_fips'].astype(str)
hpi_df['year'] = hpi_df['year'].apply(lambda x: x-1)

#merge with indicators and remove nan (not allowed in model)
df = df.merge(hpi_df, how='left', on=['year', 'county_fips'])

df.replace('.', np.nan, inplace=True)
df['hpi_change_in_a_year'] = df['hpi_change_in_a_year'].astype(float)


df = df.dropna()

#%%
# Hold back last year of data set = 2019

data = df[df['year']!=2019].reset_index(drop=True)
d2019 = df[df['year']==2019].reset_index(drop=True)

#%% split training and testing data
X_cols = ['debt_ratio_low','debt_ratio_high', 'corp_income_tax_low', 'corp_income_tax_high', 'income_tax_low', 
       'income_tax_high', 'vehicles_available','travel_time_to_work', 'population', 'household_income', 'home_value_median']

X = data[X_cols]
y = data['hpi_change_in_a_year']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#%%

regr = RandomForestRegressor(max_depth=10, random_state=0)
regr.fit(X_train, y_train)

#%%
regr.score(X_test, y_test)

# 0.4809800055081026
#%%

regr.feature_importances_

# array([0.06908273, 0.05584017, 0.05575815, 0.09441413, 0.13734019,
#        0.11378343, 0.04652843, 0.07069466, 0.09041287, 0.04431295,
#        0.22183229])

#%%

pred2019 = regr.predict(d2019[X_cols])


d2019['Predicted_HPI_change'] = pred2019

d2019['Prediction_delta'] = (d2019['hpi_change_in_a_year'] - d2019['Predicted_HPI_change'])/d2019['hpi_change_in_a_year']
d2019['Prediction_delta'].mean()
# -0.1725339522820351












