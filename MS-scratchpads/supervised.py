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
from datetime import datetime
import time

import pandas as pd
import numpy as np
import censusdata

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

import seaborn as sns

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
vehicles_per_person

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

# County births by age
query9 = """
select 
county_fips,
year,
birth_15_19_pct,
birth_20_24_pct,
birth_25_29_pct,
birth_30_34_pct,
birth_35_39_pct,
birth_40_44_pct,
birth_45_50_pct

from acs1_births_by_age
"""

births = pd.read_sql(query9, con=conn)

# County educational attainment
query10 = """
select 
county_fips,
year,
grade12_nodiploma_pct,
hs_diploma_pct,
some_college_lessthan_1yr_pct,
some_college_greaterthan_1yr_pct,
bachelor_degree_pct,
master_degree_pct,
professional_degree_pct,
doctorate_degree_pct

from acs1_educational_attainment
"""

education = pd.read_sql(query10, con=conn)

# County occupancy
query11 = """
select 
county_fips,
year,
occupied_units_pct,
vacant_units_pct

from acs1_occupancy
"""

occupancy = pd.read_sql(query11, con=conn)

# County rent
query12 = """
select 
county_fips,
year,
rent_1bed_median,
rent_2bed_median,
rent_3bed_median,
rent_4bed_median

from acs1_rent
"""

rent = pd.read_sql(query12, con=conn)

# County wage data


query13 = """
select *

from "NEW_bls_wage_by_industry"
"""

wage = pd.read_sql(query13, con=conn)

# County hpi
query14 = """
select 
county_fips,
year,
annual_change_pct

from fhfa_house_price_index
"""

hpi = pd.read_sql(query14, con=conn)

# County redfin
query15 = """
select 
*

from redfin_county_full
"""

redfin = pd.read_sql(query15, con=conn)

#%% clean up bls data
## for each industry-ownership code pull oty_avg_annual_pay_pct_chg and annual_avg_employees
# Set index for year and county fips to avoid changes to these columns
wage = wage.set_index(['year','county_fips'])
cols_to_keep = []

# loop through the columns to get a list of columns for Average Annual Pay (pct change) and Average Annual number of employees
for col in wage.columns:
    if 'oty_avg_annual_pay_pct_chg' in col:
        cols_to_keep.append(col)
    elif 'annual_avg_employees' in col:
        cols_to_keep.append(col)

# Remove columns that we want to drop and fill NAs with zero 
wage = wage[cols_to_keep]
wage = wage.fillna(0).astype(float)

# determine what columns are only zeros
zero_cols = wage.sum(axis=0).reset_index()
zero_cols = zero_cols[zero_cols[0]==0]['index'].tolist()

# Drop zero colmns and reset multi-index of year and county fips
wage = wage.drop(columns = zero_cols).reset_index(level=['year','county_fips'])

#%% clean up redfin data
# Identify the columns that we want to keep. This includes columns witih year over year percent change
cols_to_keep= ['county_fips', 'period_end', 'property_type', 'property_type_id', 'median_sale_price_yoy',
               'median_list_price_yoy', 'median_ppsf_yoy', 'median_list_ppsf_yoy', 'homes_sold_yoy',
               'pending_sales_yoy', 'new_listings_yoy', 'inventory_yoy', 'months_of_supply_yoy', 
               'median_dom_yoy', 'avg_sale_to_list_yoy', 'sold_above_list_yoy', 'price_drops_yoy',  
               'off_market_in_two_weeks_yoy']
     
# Remove columns that we want to drop
redfin = redfin[cols_to_keep]

# Create date columns for filtering.  We will baseline the year over year change using the December records.
redfin['date'] = pd.to_datetime(redfin['period_end'])
redfin['year'] = redfin['date'].dt.year
redfin['month'] = redfin['date'].dt.month
redfin = redfin[redfin['month']==12]

#%%
# find records where all numerical columns are na across the whole row and drop them
na_cols = ['median_sale_price_yoy', 'median_list_price_yoy', 'median_ppsf_yoy',
       'median_list_ppsf_yoy', 'homes_sold_yoy', 'pending_sales_yoy',
       'new_listings_yoy', 'inventory_yoy', 'months_of_supply_yoy',
       'median_dom_yoy', 'avg_sale_to_list_yoy', 'sold_above_list_yoy',
       'price_drops_yoy', 'off_market_in_two_weeks_yoy']

na_rows = redfin.index[redfin[na_cols].isnull().all(1)]
redfin = redfin.drop(index = na_rows)

# Find remaining columns that contain null values 
for col in redfin.columns:
    print(col+':', redfin[col].isnull().sum(), 'null values')

# county_fips: 0 null values
# period_end: 0 null values
# property_type: 0 null values
# property_type_id: 0 null values
# median_sale_price_yoy: 49 null values
# median_list_price_yoy: 3943 null values -> median_list_ppsf_yoy and new_listings_yoy are null as well
# median_ppsf_yoy: 543 null values
# median_list_ppsf_yoy: 4053 null values
# homes_sold_yoy: 49 null values
# pending_sales_yoy: 6817 null values
# new_listings_yoy: 4043 null values
# inventory_yoy: 1387 null values
# months_of_supply_yoy: 687 null values
# median_dom_yoy: 764 null values
# avg_sale_to_list_yoy: 638 null values
# sold_above_list_yoy: 436 null values
# price_drops_yoy: 19540 null values
# off_market_in_two_weeks_yoy: 5894 null values
# date: 0 null values
# year: 0 null values
# month: 0 null values
#%%
# Plot remaining Missing Data
import matplotlib.pyplot as plt
# fig, ax =plt.subplots(5,1)

for prop in redfin['property_type'].unique():
    temp = redfin[redfin['property_type']==prop]
    # cols = temp.set_index(['year','county_fips']).columns[:]    
    cols = temp.set_index(['county_fips','year']).columns[:]
    colors = ['#000099', '#ffff00']
    # sns.heatmap(temp.sort_values(by=['year','county_fips']).set_index(['year','county_fips'])[cols].isnull(), cmap=sns.color_palette(colors))
    sns.heatmap(temp.sort_values(by=['county_fips','year']).set_index(['county_fips','year'])[cols].isnull(), cmap=sns.color_palette(colors))

    plt.title('Property type: '+str(prop))
    plt.show()

# Suggest we drop column price_drops_yoy and only keep property_type = All Residential

redfin = redfin[redfin['property_type']=='All Residential']
redfin = redfin.drop(columns = 'price_drops_yoy')
#%%
#join datasets

df = debt.merge(corp,how='left',on=['state_fips','year']).fillna(0)
df = df.merge(tax,how='left',on=['state_fips','year']).fillna(0)
df = df.merge(vehicles,how='left',on=['county_fips','year'])
df = df.merge(travel,how='left',on=['county_fips','year'])
df = df.merge(population,how='left',on=['county_fips','year'])
df = df.merge(income,how='left',on=['county_fips','year'])
df = df.merge(home_value,how='left',on=['county_fips','year'])
df = df.merge(births,how='left',on=['county_fips','year'])
df = df.merge(education,how='left',on=['county_fips','year'])
df = df.merge(occupancy,how='left',on=['county_fips','year'])
df = df.merge(rent,how='left',on=['county_fips','year'])
df = df.merge(wage,how='left',on=['county_fips','year'])
df = df.merge(redfin,how='left',on=['county_fips','year'])

# 72211 x 184
#%%
# What years have the most populated data?
df.groupby('year').count().sum(axis=1)

# 1999     25096
# 2000     25088
# 2001     25088
# 2002     25096
# 2003     25112
# 2004     25112
# 2005     25112
# 2006     25112
# 2007     25112
# 2008     25096
# 2009     25088
# 2010     42868
# 2011     42963
# 2012     62450
# 2013     64821
# 2014    430040 *
# 2015    434590 *
# 2016    435803 *
# 2017    436238 *
# 2018    436519 *
# 2019    436757 *
# 2020    415807 *
# 2021     52480


#%%
# handle missing values
df = df[df['year'].isin([2014,2016,2017,2018,2019,2020])]
means = df.groupby(['county_fips','year','property_type']).mean()

for col in means.columns:
    print(col+':', means[col].isnull().sum(), 'null values')
# df = df.fillna(df.mean())
# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=2)

# df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

#%%
#Bring in HPI target data

#reduce year by 1 so that the hpi annual change value is for one year in the future
hpi['year'] = hpi['year'].astype(int)
hpi['county_fips'] = hpi['county_fips'].astype(str)
hpi['year'] = hpi['year'].apply(lambda x: x-1)

#merge with indicators and remove nan (not allowed in model)
df = df.merge(hpi, how='left', on=['year', 'county_fips'])

df.replace('.', np.nan, inplace=True)
df['annual_change_pct'] = df['annual_change_pct'].astype(float)


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












