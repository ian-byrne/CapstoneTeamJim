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
    
    st.write("We can fit the VAR model with all the data from 2016 to 2021 to predict the median sale price by county for 2022 and calculate the 95% confidence intervals.")
    
    summary2022prediction = pd.read_csv("streamlit/data/Summary_2022_Predictions.csv")    
    
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
    st.write("The forecasted top 10 counties to return the best real estate investment opportunities are shown below.")
    
    
    
    
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
 
    st.write("Top 10 counties with the highest percentage of growth in 2022 compared to 2021:")


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
    
    top10table.update_layout(width=1000, height=600)
    st.plotly_chart(top10table,use_container_width=True) 
   
   
    st.write("""It's interesting to see that the top 10 counties are all showing a >60% increase over 2021, with Hardin County, Ohio showing double that.  """)
      
    st.write("""Mosaic Experian reports (reference 1) provide detailed personality profiles for the typical residents of Hardin County.  The top 3 profiles are **Family Union, Families in Motion, and Singles and Starters**, and make up approximately 78% of households. Most of the households within these profiles are young families or young people starting to build their independent lives, with a high majority of the households as homeowners.  With proximity to both Columbus and Toledo, Hardin County is also great choice for more affordable housing within commuting distance.
    
    """)   
     
    
    
    st.subheader("Future Work")
    
    st.write("""Future work on this analysis could include the following ideas:  
             
*  It would be interesting to see how this model performs on the remaining U.S. counties, as well as performing a similar analysis on Canadian/International markets.  
*  Increase the granularity of the results to be able to identify sub-regions within the counties that are predicting to have higher returns.  This would require the input data to be higher granularity as well.
*  Combine or augment the models to bring more local features, like Yelp rating trends and U-haul rental trends.  
*  Investigate the impact of inflation.
 
                 
              """) 
    with st.expander("References"):
        st.write(
            """
        1.  https://www.gale.com/c/business-demographicsnow

        """ 
        )
