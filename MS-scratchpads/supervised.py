# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:07:48 2022

@author: melan
"""


import psycopg2
import secrets_melanie
import time
from datetime import datetime

import pandas as pd
import numpy as np

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

#%%
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)




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
where year >= 2014
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
where year >= 2014
"""

tax = pd.read_sql(query2, con=conn)

# County Debt ratio - long version
query3 = """
select * 
from county_debt_ratio_long
where year >= 2014
"""

debt = pd.read_sql(query3, con=conn)

# County vehicles - long version
query4 = """
select 
county_fips,
year,
vehicles_per_person

from acs1_vehicles_available
where year >= 2014
"""

vehicles = pd.read_sql(query4, con=conn)

# County travel_time- long version
query5 = """
select 
county_fips,
year,
travel_time_to_work

from acs1_travel_time_to_work
where year >= 2014
"""

travel = pd.read_sql(query5, con=conn)


# County population- long version
query6 = """
select 
county_fips,
year,
population

from acs1_population
where year >= 2014
"""

population = pd.read_sql(query6, con=conn)

# County household income- long version
query7 = """
select 
county_fips,
year,
household_income

from acs1_household_income
where year >= 2014
"""

income = pd.read_sql(query7, con=conn)

# County Median Home value- long version
query8 = """
select 
county_fips,
year,
home_value_median

from acs1_home_value_median
where year >= 2014
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
where year >= 2014
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
where year >= 2014
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
where year >= 2014
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
where year >= 2014
"""

wage = pd.read_sql(query13, con=conn)

# query13a = """
# select *

# from bls_wage_by_industry
# where year >= 2014
# """

# wage1 = pd.read_sql(query13a, con=conn)

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
county_fips,
period_end, 
property_type, 
property_type_id, 
median_sale_price_yoy,
median_list_price_yoy, 
median_ppsf_yoy, 
median_list_ppsf_yoy, 
homes_sold_yoy,
new_listings_yoy, 
inventory_yoy, 
months_of_supply_yoy, 
median_dom_yoy, 
avg_sale_to_list_yoy, 
sold_above_list_yoy



from redfin_county_full

where property_type ='All Residential'

"""

redfin = pd.read_sql(query15, con=conn)

conn.close()

#%% clean up NEW bls data
## for each industry-ownership code pull oty_avg_annual_pay_pct_chg and annual_avg_employees
# Set index for year and county fips to avoid changes to these columns
wage = wage.set_index(['year','county_fips'])
pay_cols = []
emp_cols = []


# loop through the columns to get a list of columns for Average Annual Pay (pct change) and Average Annual number of employees
for col in wage.columns:
    if 'annual_avg_employees' in col:
        emp_cols.append(col)
    elif 'avg_annual_pay' in col:
        if 'pct_chg' not in col:
            pay_cols.append(col)


# Determine the % change of employees by year, county and industry
emp = wage[emp_cols]
emp = emp.fillna(0).astype(float)
emp = emp.reset_index(level=['year','county_fips'])
# convert from wide to long format
emp = emp.melt(id_vars=['year','county_fips'],value_name = 'annual_avg_employees')
# extract industry code and ownership codes from previous columns
emp['naics_industry_code'] = emp['variable'].apply(lambda x: x[:2])
emp['owner'] = emp['variable'].apply(lambda x: x[3:4])
# as the ownership = 0 is for all ownerships, drop to avoid duplication
emp = emp[emp['owner']!='0']
# group and total the employees
emp = emp.groupby(['year','county_fips','naics_industry_code'], as_index=False)['annual_avg_employees'].sum()
emp = emp.sort_values(by=['county_fips','naics_industry_code','year'])
# shift total of employees by 1 year to access previous year's total employees and merge back onto employee df
prev_emp = emp.groupby(['county_fips','naics_industry_code'])['annual_avg_employees'].shift(1)
emp=emp.merge(prev_emp,how='left',left_index=True,right_index=True).rename(
    columns={'annual_avg_employees_x':'annual_avg_employees','annual_avg_employees_y':'previous_avg_employee'})
# due to absence of 2013 data, set previous year for 2014 to the 2014 value
emp['prev_avg_employees'] = emp['previous_avg_employee'].combine_first(emp['annual_avg_employees'])
# calculate % change
emp['avg_annual_employee_pct_chg'] = ((emp['annual_avg_employees']-emp['prev_avg_employees'])/emp['prev_avg_employees']*100).fillna(0)
emp = emp.drop(columns=['previous_avg_employee'])

### Repeat for annual pay

# Determine the % change in pay by year, county and industry
pay = wage[pay_cols]
pay = pay.fillna(0).astype(float)
pay = pay.reset_index(level=['year','county_fips'])
# convert from wide to long format
pay = pay.melt(id_vars=['year','county_fips'],value_name = 'annual_avg_pay')
# extract industry code and ownership codes from previous columns
pay['naics_industry_code'] = pay['variable'].apply(lambda x: x[:2])
pay['owner'] = pay['variable'].apply(lambda x: x[3:4])
# as the ownership = 0 is for all ownerships, drop to avoid duplication
pay = pay[pay['owner']!='0']
# group and total the pay
pay = pay.groupby(['year','county_fips','naics_industry_code'], as_index=False)['annual_avg_pay'].sum()
pay = pay.sort_values(by=['county_fips','naics_industry_code','year'])
# shift total  pay by 1 year to access previous year's total pay and merge back onto pay df
prev_pay = pay.groupby(['county_fips','naics_industry_code'])['annual_avg_pay'].shift(1)
pay=pay.merge(prev_pay,how='left',left_index=True,right_index=True).rename(
    columns={'annual_avg_pay_x':'annual_avg_pay','annual_avg_pay_y':'previous_avg_pay'})
# due to absence of 2013 data, set previous year for 2014 to the 2014 value
pay['prev_avg_pay'] = pay['previous_avg_pay'].combine_first(pay['annual_avg_pay'])
# calculate % change
pay['avg_annual_pay_pct_chg'] = ((pay['annual_avg_pay']-pay['prev_avg_pay'])/pay['prev_avg_pay']*100).fillna(0)
pay = pay.drop(columns=['previous_avg_pay'])

# combine emp and pay

wages = emp.merge(pay,how='left',on=['year','county_fips','naics_industry_code'])
# remove data where there are no employees - we will fill these values with 0 later
wages = wages[wages['annual_avg_employees']!=0]
# replace inf with 100, representing a 100% change
wages = wages.replace([np.inf,-np.inf], 100)
# industry code of 10 is a summary code for all industries 
wages=wages[wages['naics_industry_code']!='10']
# keep only percent change columns
wages = wages[['year', 'county_fips', 'naics_industry_code','avg_annual_employee_pct_chg','avg_annual_pay_pct_chg']]
# pivot to create wide version of dataset
wages = wages.pivot_table(index=['year','county_fips'], columns=['naics_industry_code']).fillna(0)
wages.columns = [str(a)+'_'+str(b) for b,a in wages.columns]
wages = wages.reset_index(['year','county_fips'])

#%% clean up redfin data
# Create date columns for filtering.  We will baseline the year over year change using the December records.
redfin['date'] = pd.to_datetime(redfin['period_end'])
redfin['year'] = redfin['date'].dt.year
redfin['month'] = redfin['date'].dt.month
redfin = redfin[redfin['month']==12]

#%%
# find records where all numerical columns are na across the whole row and drop them
na_cols = ['median_sale_price_yoy', 'median_list_price_yoy', 'median_ppsf_yoy',
       'median_list_ppsf_yoy', 'homes_sold_yoy', 'new_listings_yoy', 'inventory_yoy', 'months_of_supply_yoy',
       'median_dom_yoy', 'avg_sale_to_list_yoy', 'sold_above_list_yoy']
       

na_rows = redfin.index[redfin[na_cols].isnull().all(1)]
redfin = redfin.drop(index = na_rows)

# Find remaining columns that contain null values 
for col in redfin.columns:
    print(col+':', redfin[col].isnull().sum(), 'null values')

# county_fips: 0 null values
# period_end: 0 null values
# property_type: 0 null values
# property_type_id: 0 null values
# median_sale_price_yoy: 14 null values
# median_list_price_yoy: 672 null values
# median_ppsf_yoy: 89 null values
# median_list_ppsf_yoy: 727 null values
# homes_sold_yoy: 14 null values
# new_listings_yoy: 705 null values
# inventory_yoy: 321 null values
# months_of_supply_yoy: 98 null values
# median_dom_yoy: 238 null values
# avg_sale_to_list_yoy: 188 null values
# sold_above_list_yoy: 140 null values
# date: 0 null values
# year: 0 null values
# month: 0 null values

# redfin.county_fips.nunique()
# 1625


#%%
# Use KNN imputer to impute
imputer = KNNImputer(n_neighbors=2)

redfin = redfin.set_index(['year','county_fips'])
redfin_imputed = pd.DataFrame(imputer.fit_transform(redfin[na_cols]), columns=na_cols)
redfin_imputed.index=redfin.index
redfin_imputed = redfin_imputed.reset_index(['year','county_fips'])
#%%
#join datasets
# debt #1999-2021
df = debt.merge(corp,how='left',on=['state_fips','year']) #2014-2022
df = df.merge(tax,how='left',on=['state_fips','year']) #2014-2022
df = df.merge(wages,how='left',on=['county_fips','year']).fillna(0) #2014-2020
df = df.merge(redfin_imputed,how='inner',on=['county_fips','year']) #2012-2021


#%%
# print("Number of unique county fips: "+str(df.county_fips.nunique()))
# for col in df.columns:
#     print(col+':', df[col].isnull().sum(), 'null values')

#%%
df = df.merge(vehicles,how='inner',on=['county_fips','year']) #2014-2019
df = df.merge(travel,how='left',on=['county_fips','year']) #2014-2019
df = df.merge(population,how='left',on=['county_fips','year']) #2014-2019
df = df.merge(income,how='left',on=['county_fips','year']) #2014-2019
df = df.merge(home_value,how='left',on=['county_fips','year']) #2014-2019
df = df.merge(births,how='left',on=['county_fips','year']) #2014-2019
df = df.merge(education,how='left',on=['county_fips','year']) #2014-2019
df = df.merge(occupancy,how='left',on=['county_fips','year']) #2014-2019
# df = df.merge(rent,how='left',on=['county_fips','year']) #2015-2019

# df.county_fips.nunique()
# 632

# 3606x 78

#%%
# Plot null values in data

# # for prop in df['property_type'].unique():
# # temp = df[df['property_type']==prop]
# cols = df.set_index(['year','county_fips']).columns[:]    
# # cols = df.set_index(['county_fips','year']).columns[:]
# colors = ['#000099', '#ffff00']
# sns.heatmap(df.sort_values(by=['year','county_fips']).set_index(['year','county_fips'])[cols].isnull(), cmap=sns.color_palette(colors))
# # sns.heatmap(df.sort_values(by=['county_fips','year']).set_index(['county_fips','year'])[cols].isnull(), cmap=sns.color_palette(colors))

# # plt.title('Property type: '+str(prop))
# plt.show()

#%%
# handle missing values

df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df_imputed['year'] = df_imputed['year'].astype(int)
df_imputed['county_fips'] = df_imputed['county_fips'].astype(int).astype(str)
df_imputed['county_fips'] =df_imputed['county_fips'].str.zfill(5) 

#%%
#Bring in HPI target data

#reduce year by 1 so that the hpi annual change value is for one year in the future
hpi['year'] = hpi['year'].astype(int)
hpi['county_fips'] = hpi['county_fips'].astype(str)
hpi['year'] = hpi['year'].apply(lambda x: x-1)

#merge with indicators and remove nan (not allowed in model)
df_imputed = df_imputed.merge(hpi, how='left', on=['year', 'county_fips'])


df_imputed['annual_change_pct'] = df_imputed['annual_change_pct'].astype(float)

df_imputed = df_imputed.reset_index(drop=True)
df_imputed = df_imputed.dropna()

#%%
# Hold back last year of data set = 2019

data = df_imputed[df_imputed['year']!=2019].reset_index(drop=True)
d2019 = df_imputed[df_imputed['year']==2019].reset_index(drop=True)

#%% split training and testing data
X = data.iloc[:,3:-1]
y = data.iloc[:,-1]

#Normalize dataset
X_norm = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.25, random_state=42)

#%%
# RF regressor instantiate and train
regr = RandomForestRegressor(max_depth=10, random_state=0)
regr.fit(X_train, y_train)

#%%
score = regr.score(X_test, y_test)
print("Random Forest Regressor Test score: "+str(score)+'\n')
# 0.523631453593925
#%%

top_feature_indices = regr.feature_importances_.argsort()[::-1]
top10_features = d2019.columns[top_feature_indices][0:10]
print("Top 10 features:")
for i,feature in enumerate(top10_features):
    print("\t{} ({:.2f})".format(feature,regr.feature_importances_[top_feature_indices[i]]))

#%%

d2019_norm = StandardScaler().fit_transform(d2019.iloc[:,3:-1])
pred2019 = regr.predict(d2019_norm)


d2019['Predicted_HPI_change'] = pred2019

d2019['Prediction_delta'] = ((d2019['annual_change_pct'] - d2019['Predicted_HPI_change'])/d2019['annual_change_pct'])*100
d2019['Prediction_delta'].mean()
# 6.93%

#%%
##  Plot deltas to see if geographical trend

fig = px.choropleth(d2019, geojson=counties, locations='county_fips', color='Prediction_delta',
                           color_continuous_scale="Viridis",
                            range_color=(0, 100),
                           scope="usa",
                           labels={'Prediction_delta':'Prediction delta for 2019 HPI'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


#%%

####################################      AUTO ML REGRESSOR       ####################################  
# Cannot be run on Windows machine
import autosklearn.regression
from pprint import pprint

# Instantiate the Regressor
automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='/tmp/autosklearn_regression_example_tmp',
)

# fit regressor using same normalized train/test split
automl.fit(X_train, y_train, dataset_name='X_norm')

#%%
# pretty print top results
pprint(automl.leaderboard())


pprint(automl.show_models(), indent=4)

#%%

train_predictions = automl.predict(X_train)
print("Train R2 score:", sklearn.metrics.r2_score(y_train, train_predictions))
test_predictions = automl.predict(X_test)
print("Test R2 score:", sklearn.metrics.r2_score(y_test, test_predictions))

plt.scatter(train_predictions, y_train, label="Train samples", c='#d95f02')
plt.scatter(test_predictions, y_test, label="Test samples", c='#7570b3')
plt.xlabel("Predicted value")
plt.ylabel("True value")
plt.legend()
plt.plot([30, 400], [30, 400], c='k', zorder=0)
plt.xlim([30, 400])
plt.ylim([30, 400])
plt.tight_layout()
plt.show()






