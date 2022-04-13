"""Example Model file two."""
import streamlit as st




import pandas as pd
import os

from urllib.request import urlopen
import json

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
#%%

def model2():
    st.write("Model 2 running...Melanie Testing Streamlit interface - here is the Time series analysis")

    
    
    st.header("Vector Autoregression (VAR)  :clock4: 	")
    
    st.subheader("VAR Prediction Error  :dart: 	")    
    
    st.write("Below is the histogram of residual error")
    
    pred = pd.read_csv("data/Data_hist_scatter_2020Predictions.csv")    
    
    hist = px.histogram(pred['log_pred_errors'],title='Histogram of residual error of model')       
    # Pred2020_error = px.choropleth(Predictions2020, geojson=counties, locations='FIPS', color='Forecast error %',
    #                         color_continuous_scale="Viridis",
    #                         hover_name = 'County',
    #                         hover_data =['Predicted Median Sale Price 2020'],
    #                         scope="usa",
    #                         labels={'Forecast error %':'2020 % Forecast error'},
    #                         title = 'Average forecast error by ACS county for 2020 prediction, with most counties having less error than 5%'
    #                       )
    
    st.plotly_chart(hist,use_container_width=True)
    
    
    st.write("Below is the scatter of residual error")
    
       
    pred_errors = px.scatter(pred,x='log_median_sale_price',y='log_pred_errors', title='Residual error clustered around middle of plot with tight range and no clear patterns')
    
    st.plotly_chart(pred_errors,use_container_width=True)



    st.write("Below is the predictions error for all counties")
    
    Predictions2020 = pd.read_csv("data/Prediction_error_2020.csv")    
    
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)
        
    Pred2020_error = px.choropleth(Predictions2020, geojson=counties, locations='FIPS', color='Forecast error %',
                            color_continuous_scale="Viridis",
                            hover_name = 'County',
                            hover_data =['Predicted Median Sale Price 2020'],
                            scope="usa",
                            labels={'Forecast error %':'2020 % Forecast error'},
                            title = 'Average forecast error by ACS county for 2020 prediction, with most counties having less error than 5%'
                          )
    
    st.plotly_chart(Pred2020_error,use_container_width=True)


    st.header("Real Estate Predictions from VAR model  :clock8:")
    st.write("Below is the Median sales price prediction for all ACS counties for 2020")

    summary2020prediction = pd.read_csv("data/Summary_2020_Predictions.csv")    
    
        
    forecasted_pct = px.choropleth(summary2020prediction, geojson=counties, locations='FIPS', color='Median Sale Price % increase',
                               color_continuous_scale="Viridis",
                                # range_color=(0, 100),
                               hover_name = 'County',
                               hover_data =['Predicted Median Sale Price 2020','Lower 95% Prediction Inverval','Upper 95% Prediction Inverval'],
                               scope="usa",
                               labels={'Mean_pred_price_pct':'% price change from 2019'},
                               title='All 613 ACS counties - 2020 average Median Sale Price % increase over 2019'
                              )
    
    st.plotly_chart(forecasted_pct,use_container_width=True)
    
    
    st.header("Top 10 Counties from VAR model  :clock10:")
    st.write("Below is the top 10 predicted counties for model comparison")
    
    
    
    
    top_cnt = px.choropleth(summary2020prediction[0:10], geojson=counties, locations='FIPS', color='Median Sale Price % increase',
                               color_continuous_scale="Viridis",
                                # range_color=(0, 100),
                               hover_name = 'County',
                               hover_data =['Predicted Median Sale Price 2020','Lower 95% Prediction Inverval','Upper 95% Prediction Inverval'],
                               scope="usa",
                               labels={'Mean_pred_price_pct':'% price change from 2019'},
                               title='Top 10 ACS counties - 2020 average Median Sale Price % increase over 2019'
                              )
    
    st.plotly_chart(top_cnt,use_container_width=True)
