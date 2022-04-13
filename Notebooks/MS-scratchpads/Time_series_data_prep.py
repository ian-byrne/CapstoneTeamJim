# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 21:37:22 2022

@author: melan
"""

# Time series Data preparation

import psycopg2
import secrets_melanie
import pandas as pd
import numpy as np

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

# Redfin 
query15 = """
select 
county_fips,
region,
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

#%%
# clean up redfin data
# Create date columns for filtering.  We will baseline the year over year change using the December records.
redfin = redfin_df.copy(deep=True)
redfin['date'] = pd.to_datetime(redfin['period_end'])
redfin['year'] = redfin['date'].dt.year
redfin['month'] = redfin['date'].dt.month

county_name = redfin[['county_fips','region']].dropna().drop_duplicates()
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

# Using redfin median sale price for time series
df = base.copy(deep=True)
df['day']=1
# time series needs datetime for index
df['date'] = pd.to_datetime(df[['year','month','day']])
df['year'] = df['year'].astype(str)
df = df[['county_fips','date','median_sale_price','year']]

# Determine how many counties have missing values in training df and how many have recent missing (NOTE: some counties have no missing data so they
# will not be included in missing_vals df)

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
# if recent % of values are less than 10% then don't filter these ones out
missing_vals['to_filter'][((missing_vals['recent_pct']<10))]=0
missing_counties = missing_vals[missing_vals['to_filter']==1]['county_fips'].to_list()
# dropping 467 counties with recent vals <10% in last 6 years.
keep_counties = df[~df['county_fips'].isin(missing_counties)]['county_fips'].drop_duplicates()
# exporting VAR counties to csv for equal comparison against other models
keep_counties.to_csv("C:/Users/melan/repo/Capstone/CapstoneTeamJim/MS-scratchpads/VAR_counties.txt",index=False)


####  For 2022 predictions use all counties that have gaps in the data <10% in the last 6 years 
keep_counties = keep_counties.reset_index()
keep_counties['county_fips'] = keep_counties['county_fips'].astype(str)
    
# remove counties with sparse data and fill forward na for median sale price and backfill any values that didn't start in Jan 2016
df = df[df['county_fips'].isin(keep_counties['county_fips'])]
df['median_sale_price'] = df.groupby('county_fips')['median_sale_price'].transform(lambda v: v.ffill()).fillna(method="bfill")
# Predicting on log of median sale price
df['log_median_sale_price'] = np.log(df['median_sale_price'])
# split training and testing data and pivot so counties are columns (only using data from 2016 onward)
df = df[((df['year']>='2016'))]
df = df.merge(county_name,how='left',on='county_fips')
df.to_pickle("C:/Users/melan/repo/Capstone/CapstoneTeamJim/MS-scratchpads/data_log_2016_2021_VARcountysubset.pkl")
