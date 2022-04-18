import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor

import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

def tree_model():
    ## read data 

    df = pd.read_pickle("data/processed/fulldataset.pkl")
    df = df.sort_values(by=['county_fips', 'year']).reset_index(drop=True)

    ## isolate target column

    medsale = df[['county_fips', 'year','median_sale_price']]
    df = df.drop(['annual_change_pct', 'median_sale_price'], axis=1)

    ## reduce year on target column to predict next year

    medsale['year'] = medsale['year'] - 1
    df = df.merge(medsale, on=['year', 'county_fips'], how='left')
    df = df.dropna()

    ## only use counties available in other models

    with open("data/processed/VAR_counties.txt", 'r') as f:
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

    test_score = round(et.score(X_test, y_test), 3)
    val_score = round(et.score(X_val, y_val), 3)

    ## chart of feature importances

    d = {'Feature': X.columns, 'Importance': et.feature_importances_}
    feature_df = pd.DataFrame(d)
    feature_df = feature_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    feature_df.to_csv("streamlit/data/tree_model1_features.csv")

    fig = px.bar_polar(feature_df.iloc[:30,:], r='Importance', theta='Feature',
                color='Feature', template='plotly_dark',
                color_discrete_sequence=px.colors.sequential.Plasma_r)

    fig.update_layout(
                  height = 800,
                  width = 1500,
                  font=dict(size = 18),
                  margin = {'t':50, 'b':50, 'l':50, 'r':50})

    fig.write_image("reports/figures/Supervised/tree_model1_feature_importances.png")

    ## chart of prediction delta

    pred2019 = et.predict(X_val)
    data2019['Predicted_sale_price_change'] = pred2019
    data2019['Prediction_delta'] = ((data2019['median_sale_price'] - data2019['Predicted_sale_price_change'])/data2019['median_sale_price'])*100
    ##TODO rename columns for viz
    fig = px.choropleth(data2019, geojson=counties, locations='county_fips', color='Prediction_delta',
                            color_continuous_scale="Viridis",
                                range_color=(0, 100),
                            hover_name = 'county_fips',
                            hover_data =['Predicted_sale_price_change','median_sale_price'],
                            scope="usa",
                            labels={'Prediction_delta':'% Difference Predicted-Actual'},
                            title='Prediction Delta 2019 Predicting 2020 to Actual 2020 Values'
                            )

    fig.write_image("reports/figures/Supervised/tree_model1_prediciton_delta.png",width=1980, height=1080)

    ## calculate rmse

    rmse = '$'+str(round(mean_squared_error(data2019['median_sale_price'], data2019['Predicted_sale_price_change'], squared=False)))


    fig = go.Figure(data=[go.Table(header=dict(values=['Test R2', '2019 Validation R2', '2019 RMSE']),
                    cells=dict(values=[[test_score], [val_score], [rmse]]))
                        ])
    fig.write_image("reports/figures/Supervised/tree_model1_scores.png", width=500, height=300)

    corr_df = df[['home_value_median', 'median_ppsf', 'median_list_price', 'median_list_ppsf', 'median_sale_price']]
    corr_df.to_csv("streamlit/data/correlation_matrix.csv")
    mask = np.triu(np.ones_like(corr_df.corr(), dtype=np.bool))
    sns.heatmap(corr_df.corr(), mask=mask, annot=True, cmap='RdPu')
    plt.savefig('reports/figures/Supervised/tree_model1_correlation_matrix.png', width=500, height=300, bbox_inches="tight")

    ## model2 with highly correlated features removed

    df = df.drop(['home_value_median', 'median_list_price', 'median_ppsf', 'median_list_ppsf'], axis=1)
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

    test_score = round(et.score(X_test, y_test), 3)
    val_score = round(et.score(X_val, y_val), 3)

    ## chart of feature importances

    d = {'Feature': X.columns, 'Importance': et.feature_importances_}
    feature_df = pd.DataFrame(d)
    feature_df = feature_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    feature_df.to_csv("streamlit/data/tree_model2_features.csv")

    fig = px.bar_polar(feature_df.iloc[:30,:], r='Importance', theta='Feature',
                color='Feature', template='plotly_dark',
                color_discrete_sequence=px.colors.sequential.Plasma_r)

    fig.update_layout(
                  height = 800,
                  width = 1500,
                  font=dict(size = 18),
                  margin = {'t':50, 'b':50, 'l':50, 'r':50})

    fig.write_image("reports/figures/Supervised/tree_model2_feature_importances.png")

    ## chart of prediction delta

    pred2019 = et.predict(X_val)
    data2019['Predicted_sale_price_change'] = pred2019
    data2019['Prediction_delta'] = ((data2019['median_sale_price'] - data2019['Predicted_sale_price_change'])/data2019['median_sale_price'])*100
    data2019 = data2019[['county_fips', 'median_sale_price', 'Predicted_sale_price_change', 'Prediction_delta']]
    data2019['percent_error'] = np.absolute(data2019['Prediction_delta'])/data2019['median_sale_price']*100
    data2019.columns = ['FIPS', 'Median Sale Price 2020', 'Predicted', 'error', '2020 % Forecast Error']
    county_names = pd.read_csv('data/processed/county_names.csv', header=1)
    data2019['FIPS'] = data2019['FIPS'].apply(lambda x: int(x))
    county_names['FIPS'] = county_names['FIPS'].apply(lambda x: int(x))
    county_names['County'] = county_names['name'].astype(str) + ', ' + county_names['state'].astype(str)

    data2019 = data2019.merge(county_names, how='left', on='FIPS')
    data2019['Predicted Median Sale Price 2020'] = data2019['Predicted'].apply(lambda x: "${:,.0f}".format(x))
    data2019['2020 % Forecast Error'] = data2019['2020 % Forecast Error'].apply(lambda x: round(x, 3))
    data2019.to_csv("streamlit/data/prediction_map.csv", index=False)

    fig = px.choropleth(data2019, geojson=counties, locations='FIPS', color='2020 % Forecast Error',
                            color_continuous_scale="Viridis",
                                range_color=(0, 100),
                            hover_name = 'County',
                            hover_data =['Predicted Median Sale Price 2020'],
                            scope="usa",
                            labels={'Forecast error %':'2020 % Forecast error'},
                            title = 'Average forecast error by ACS county for 2020 prediction'
                            )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.write_image("reports/figures/Supervised/tree_model2_prediciton_delta.png", width=500, height=300)

    ## calculate rmse

    rmse = '$'+str(round(mean_squared_error(data2019['Median Sale Price 2020'], data2019['Predicted'], squared=False)))


    fig = go.Figure(data=[go.Table(header=dict(values=['Test R2', '2019 Validation R2', '2019 RMSE']),
                    cells=dict(values=[[test_score], [val_score], [rmse]]))
                        ])
    fig.write_image("reports/figures/Supervised/tree_model2_scores.png", width=500, height=300)

    print("All models and images have been exported")

if __name__ == "__main__":
    tree_model()
