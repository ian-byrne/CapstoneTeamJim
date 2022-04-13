"""Example Model file two."""
import streamlit as st


from config import definitions
root_dir = definitions.root_directory()

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
    print("Model 2 demo file")
    
    st.write("Below is the predictions for all the available counties")
    
    summary2022prediction = pd.read_csv(os.path.join(root_dir,"..","reports","results","Time_Series","Summary_2022_Predictions_.csv"))
    
    
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
    
    Pred2022_plot.show()
    Pred2022_plot.write_image(os.path.join(root_dir,"..","reports","figures","Time_Series","Choro_Summary_all_counties2022_predictions.png"),width=1980, height=1080)
    
    
    
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
    
    Top_10_Pred2022_plot.show()
    Top_10_Pred2022_plot.write_image(os.path.join(root_dir,"..","reports","figures","Time_Series","Choro_Summary_top10_counties_2022_predictions.png"),width=1980, height=1080)