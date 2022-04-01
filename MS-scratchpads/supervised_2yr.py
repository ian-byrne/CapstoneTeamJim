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
from sklearn.ensemble import ExtraTreesRegressor
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
# secrets = secrets_melanie.secrets()

# conn = psycopg2.connect(host=secrets['db_url'],
#         port=secrets['port'],
#         dbname=secrets['db_name'],
#         user=secrets['username'],
#         password=secrets['password'],
#         connect_timeout=10)

# cur = conn.cursor()
#%%
# Bring in datasets
data = pd.read_pickle('C:/Users/melan/repo/Capstone/CapstoneTeamJim/MS-scratchpads/dataraw_to2018.pkl')

with open('C:/Users/melan/repo/Capstone/CapstoneTeamJim/JN-scratchpads/VAR_counties.txt', 'r') as f:
    lines = f.readlines()

VAR_counties = []
for line in lines:
    VAR_counties.append(line.strip())
    
data = data[data['county_fips'].isin(VAR_counties)]
# 620 counties remaining


dtar = pd.read_pickle('C:/Users/melan/repo/Capstone/CapstoneTeamJim/MS-scratchpads/dataraw_2019.pkl')
dtar = dtar[dtar['county_fips'].isin(VAR_counties)]

target_cols = ['annual_change_pct']
new_columns = data.columns.drop(target_cols).tolist() + target_cols

data = data[new_columns]
dtar = dtar[new_columns]

#%% split training and testing data
X = data.iloc[:,3:-1]
y = data.iloc[:,-1]

#Normalize dataset
# X_norm = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#%%
# Extra Trees regressor instantiate and train
et = ExtraTreesRegressor(max_depth=15, n_estimators=500, random_state=0).fit(
       X_train, y_train)


#%%
score = et.score(X_test, y_test)
print("Extra Trees Regressor Test score: "+str(score)+'\n')
# 0.671 using 2018 and 2019 data
# 0.243 using 2019 and 2020 data
#%%

top_feature_indices = et.feature_importances_.argsort()[::-1]
top10_features = X.columns[top_feature_indices][0:10]
print("Top 10 features:")
for i,feature in enumerate(top10_features):
    print("\t{} ({:.2f})".format(feature,et.feature_importances_[top_feature_indices[i]]))

#%%
et.fit(X,y)


pred = et.predict(dtar.iloc[:,3:-1])


dtar['Predicted_HPI_change'] = pred
# mean_absolute_error(d2020['annual_change_pct'], d2020['Predicted_HPI_change'])
#1.68



dtar['Prediction_error'] = (dtar['annual_change_pct'] - dtar['Predicted_HPI_change'])#/d2019['annual_change_pct']*100
dtar['annual_change_pct'].mean()
# 4.163
dtar['Predicted_HPI_change'].mean()
# 5.43

dtar['Prediction_error'].mean()
# -1.275

dtar['Prediction_error'].std()
# 1.69



hist = px.histogram(dtar['Prediction_error'])
hist.show()

#%%
##  Plot deltas to see if geographical trend

fig = px.choropleth(dtar, geojson=counties, locations='county_fips', color='Prediction_delta',
                           color_continuous_scale="Viridis",
                            range_color=(-4, 4),
                           scope="usa",
                           labels={'Prediction_delta':'Prediction delta for 2019 HPI'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
#%%






