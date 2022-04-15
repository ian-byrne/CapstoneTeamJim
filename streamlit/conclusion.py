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
    
    st.write("We can fit all the data from 2016 to 2021 and predict the median sale price by county for 2022 and calculate the 95% confidence intervals.")
    
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
    
    Pred2022_plot.update_layout(height = 500)      
    st.plotly_chart(Pred2022_plot,use_container_width=True)


    st.write(""" #           
             
              """) 
              
    st.write(""" #           
             
              """)               
    
    st.header("2022 Top 10 Counties for Real Estate Investment Opportunities  :heavy_dollar_sign:")
    st.write("The forecasted top 10 counties to return the best real estate investment opportunities are shown below")
    
    
    
    
    Top_10_Pred2022_plot = px.choropleth(summary2022prediction[0:10], geojson=counties, locations='FIPS', color='Median Sale Price % increase',
                               color_continuous_scale="Viridis",
                                # range_color=(0, 50),
                               hover_name = 'County',
                               hover_data =['Predicted Median Sale Price 2022','Lower 95% Prediction Inverval','Upper 95% Prediction Inverval'],
                               scope="usa",
                               labels={'Median Sale Price % increase':'% price change from 2021'},
                               title = 'Top 10 Counties - 2022 average Median Sale Price % increase over 2021'
                              )
    Top_10_Pred2022_plot.update_layout(height = 500)     
    st.plotly_chart(Top_10_Pred2022_plot,use_container_width=True)
 
    st.write(""" #           
             
              """) 
              
    st.write(""" #           
             
              """)     
 
    st.write("Top 10 counties as having the highest percentage of growth in 2022 compared to 2021:")


    top10table = go.Figure(data=[go.Table(
                header=dict(values=['County','2022 Price % increase over 2021'],
                line_color='darkslategray',
                fill_color='#3366CC',
                align=['center','center'],
                font=dict(color='white', size=20),
                height=40,),
                cells=dict(values=[summary2022prediction[0:10]['County'], summary2022prediction[0:10]['Median Sale Price % increase']],
                line_color='darkslategray',
                fill=dict(color=['white', 'white']),
                align=['center', 'center'],
                font_size=18,
                height=30)
                  )])
    
    top10table.update_layout(width=1000, height=700)
    st.plotly_chart(top10table,use_container_width=True) 
    
    
    st.subheader("Future Work")
    
    st.write("""This chart provides the following information:  
             
*  It would be interesting to see how this model performs on the remaining US counties, as well as performing a similar analysis on Canadian/International markets.  
*  Can the models be combined or improved to bring more local features
*  With all the recent commentary on inflation, we could in a future iteration use that data as well.
 
                 
              """) 
