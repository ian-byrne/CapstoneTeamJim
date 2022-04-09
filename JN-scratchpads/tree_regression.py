import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
# render plot in default browser
pio.renderers.default = 'browser'

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


## read data 

df = pd.read_pickle('fulldataset.pkl')
df = df.sort_values(by=['county_fips', 'year']).reset_index(drop=True)

## isolate target column

medsale = df[['county_fips', 'year','median_sale_price']]
df = df.drop(['annual_change_pct', 'median_sale_price'], axis=1)

## reduce year on target column to predict next year

medsale['year'] = medsale['year'] - 1
df = df.merge(medsale, on=['year', 'county_fips'], how='left')
df = df.dropna()

## only use counties available in other models

with open('VAR_counties.txt', 'r') as f:
    lines = f.readlines()

VAR_counties = []
for line in lines:
    VAR_counties.append(line.strip())

df = df[df['county_fips'].isin(VAR_counties)]

## create train/test/val

data = df[df['year']!=2019]
data2019 = df[df['year']==2019]


X = data.iloc[:,3:-1]
y = data.iloc[:,-1]

X_val = data2019.iloc[:,3:-1]
y_val = data2019.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

## fit best model determined by pyCaret and gridsearch

et = ExtraTreesRegressor(max_depth=15, n_estimators=500, random_state=0).fit(
       X_train, y_train)

filename = 'tree_model1.pkl'
pickle.dump(et, open(filename, 'wb'))

test_score = et.score(X_test, y_test)
val_score = et.score(X_val, y_val)

## TODO add charts to show test and validation scores

## chart of feature importances

d = {'Feature': X.columns, 'Importance': et.feature_importances_}
df = pd.DataFrame(d)
df = df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

fig = px.bar_polar(df.iloc[:30,:], r='Importance', theta='Feature',
            color='Feature', template='plotly_dark',
            color_discrete_sequence=px.colors.sequential.Plasma_r)

fig.write_image("tree_model1_feature_importances.png")

## chart of prediction delta

pred2019 = et.predict(X_val)
data2019['Predicted_sale_price_change'] = pred2019
data2019['Prediction_delta'] = ((data2019['median_sale_price'] - data2019['Predicted_sale_price_change'])/data2019['median_sale_price'])*100

fig = px.choropleth(data2019, geojson=counties, locations='county_fips', color='Prediction_delta',
                           color_continuous_scale="Viridis",
                            range_color=(0, 100),
                           scope="usa",
                           labels={'Prediction_delta':'Prediction delta for 2019 HPI'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.write_image("tree_model1_prediciton_delta.png")

## calculate rmse

rmse = mean_squared_error(data2019['median_sale_price'], data2019['Predicted_sale_price_change'], squared=False)

## TODO add correlation plot and model removing highly correlated features

print('Model 1 Test Score:', test_score, 
        'Model 1 Validation Score:', val_score, 
        'Model 1 RMSE:', rmse)
