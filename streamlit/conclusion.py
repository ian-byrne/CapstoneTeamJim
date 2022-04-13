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

def conclusion():
   
    
    st.header("2022 Real Estate Price change from 2021  :house_with_garden: 	")
    
    st.write("Below is the predictions for all the available counties")
    
    summary2022prediction = pd.read_csv("data/Summary_2022_Predictions.csv")    
    
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)
        
    Pred2022_plot = px.choropleth(summary2022prediction, geojson=counties, locations='FIPS', color='Median Sale Price % increase',
                               color_continuous_scale="Viridis",
                                # range_color=(0, 50),
                               hover_name = 'County',
                               hover_data =['Predicted Median Sale Price 2022','Lower 95% Prediction Inverval','Upper 95% Prediction Inverval'],
                               scope="usa",
                               labels={'Median Sale Price % increase':'% price change from 2021'},
                               title = 'All counties - 2022 average Median Sale Price % increase over 2021'
                              )
    
    st.plotly_chart(Pred2022_plot,use_container_width=True)
    
    
    st.header("2022 Top 10 Counties for Real Estate Investment Opportunities  :heavy_dollar_sign:")
    st.write("Below is the top 10 predicted counties for investment opportunities in 2022")
    
    
    
    
    Top_10_Pred2022_plot = px.choropleth(summary2022prediction[0:10], geojson=counties, locations='FIPS', color='Median Sale Price % increase',
                               color_continuous_scale="Viridis",
                                # range_color=(0, 50),
                               hover_name = 'County',
                               hover_data =['Predicted Median Sale Price 2022','Lower 95% Prediction Inverval','Upper 95% Prediction Inverval'],
                               scope="usa",
                               labels={'Median Sale Price % increase':'% price change from 2021'},
                               title = 'Top 10 Counties - 2022 average Median Sale Price % increase over 2021'
                              )
    
    st.plotly_chart(Top_10_Pred2022_plot,use_container_width=True)
