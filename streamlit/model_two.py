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
    st.header("Vector Autoregression (VAR)  :clock4: 	")
        
        
    st.write("""
             Vector Autoregression (VAR) is a multivariate time series model that extracts the relationship between datasets over time.  In this project, we explore the relationship between median sales prices between US counties over time.  How does the median sales price for county X influence its neighboring county Y?  Being able to establish the relationship between the counties, we are able to use this information to support the prediction of future median sales prices vs. a univariate time series model that would predict the sale price by county on its own.
            
            
            """)

    st.write("""
              The data used in this model is the redfin monthly data.  We have data from 2012 to 2021 but there are missing values (not all counties have sales data every month since 2012).  The number of unique counties available in the redfin data is 1860.
              
              
              """)   

    st.write("""
             In order to determine the extent of the missing values, we create a base dataframe that includes all months from 2012 to 2021 for all counties and left merge the redfin data.  From there we can investigate the null values.  We want to know what is the percentage of missing data by county for the whole time period and what is the percentage of missing data for recent time periods (ex. 2016 onward).  We find that 467 counties have missing data >10% of the whole dataset.  We decide to drop these counties to improve the model accuracy and are left with 1393 counties. 
          
             
              """)     

    st.write("""
             In order to compare the results between models we need to ensure that we are using the same counties.  Since ACS demographic data contains data at the county level for county populations > 65,000 we need to find the subset of counties that are available between both datasets.  This reduces the final dataset to 613 counties.
   
             
              """) 


    st.write("""
             As we still have missing data, we do a forward fill at the county level.  This will ensure that the median sales price trend for the missing months remains constant based on the previous month.  Finally, we do a log transform on the median sales price.  We use a log transformation to remove the skew of the data as there is quite a variance between sale prices in different areas.
             
             
              """) 


    st.write("""
             We split the training and testing data, reserving all data from 2020 for the holdout dataset.  The data is pivoted to have each county represented as a vector (column) and have the monthly data as the rows. 
             
             
              """) 

    st.write(""" #           
             
              """)     



    st.image("streamlit/data/model_2_df.png", caption="Dataset showig the first year of values", width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")


    st.write(""" #           
             
              """)  
              
    st.write("""
             We can test if the sales price of county X has a causal relationship with county Y’s sale price by performing the Granger Causality test.  This test is another statistical test that returns a p-value to test the null hypothesis that there is no relationship between the series.  If the p-value is less than 0.05 we reject the null hypothesis to conclude that county X has a causal relationship with county Y.  As we performed this test on 1393 different counties, it took over 4 hours to run and the results did return several causal relationships.
             
             
              """) 


    st.write("""
             Next, for a time series analysis we need to test if the time series is stationary or not.  The Augmented Dicky Fuller test (ADF test) is a hypothesis test to determine if a time series is stationary.  If the p-value is less than 0.05, the time series is stationary.  If the p-value is greater than 0.05, the time series is not stationary.  But we can make the time series stationary by differencing and re-running the ADF test.  In the case of the redfin data, we need to difference the series twice before all the series are stationary             
             
             
              """)
              
    st.write("""
             Now we are ready for modeling.  The VAR model accepts a few important parameters, specifically max_lags and frequency.  We set the frequency to monthly to reflect our data and the max_lags to 12.  We tested a few different models with different max_lags but 12 returned the best results.
             
             
              """) 
              
    st.write(""" #           
             
              """)  
    code = """  
                                
# instantiate the variables   
num_forecasts = 12  
var_res, forecasts = None, None  

# create the model using second differencing and set the frequency to Monthly
model = VAR(df_2diff_log,freq='MS')
# fit the model using 12 maxlags
var_res = model.fit(maxlags=12)

# predict the next 12 months
forecasts = var_res.forecast(df_2diff_log.values,steps=num_forecasts)           
            """          
    st.code(code,language = 'python')

    st.write(""" #           
             
              """)  

              
    st.write("""
             We need to “undifference” the forecasts - twice - in order to get the forecasts back to the original scale so we can compare the forecasted values for 2020 against the held out testing data set.              
             
             
              """) 

    
    st.subheader("VAR Prediction Error  :dart: 	")    
    
    st.write("""
             We can plot the residual errors on a histogram to visualize the distribution.  As you can see in the chart below, the residual errors have a zero mean and are normally distributed and will support our calculation for prediction intervals. ")
             
             
              """) 
              
    pred = pd.read_csv("streamlit/data/Data_hist_scatter_2020Predictions.csv")    
    
    hist = px.histogram(pred['log_pred_errors'],
                        title='Histogram of residual error of model',
                        labels={"value": "Prediction Errors"},
                        color_discrete_map={"log_pred_errors": "#3366CC"},
                        template = "plotly_white"
                        )
    hist.update_layout(showlegend=False)
    
    st.plotly_chart(hist,use_container_width=True)
    
    
    st.write("""
             We can examine residuals by plotting the residual errors against the predicted values as shown in the scatter plot below.              
             
              """) 

    st.write("""This chart provides the following information:  
             
*  The cluster is towards the middle of the plot and it’s symmetrical about the 0 axis    
*  The cluster is condensed and not spread out across the y axis   
*  There are no clear patterns  

This tells us that the model is learning very well. 
                 
              """) 
       
    pred_errors = px.scatter(pred,x='log_median_sale_price',y='log_pred_errors', 
                             title='Residual error plot',
                             labels={"log_median_sale_price": "True Median Sale Price (log)",
                                     "log_pred_errors": "Predicted errors (log)"},

                             template = "plotly_white"
                        )
    
    pred_errors.update_traces(marker_color="#3366CC")
    st.plotly_chart(pred_errors,use_container_width=True)



    st.write("""
             But our values are still in log transform.  That won’t be very useful to the user so we need to transform it back into dollars.  This is not as simple as exponentiating the values as we need to account for the residual errors.  We can account for the residual error of the log transformed model by multiplying the exponentiated log prediction with the mean of the exponentiated prediction errors, as shown in the formula below:            
           
                 
             
           """) 

 

 
           
    st.latex(r""" 
             
             \hat{Y}\jmath = exp(\widehat{lnY\jmath})\cdot \frac{1}{N}\sum_{\imath=1}^{N}exp(e\imath)
             
             
             
             """)           

    st.write(""" #           
             
              """)  


    st.write("""
             
             
             The table below shows the RMSE and R squared score of the model for the 613 counties subset.  We can compare these metrics against the other models. 
                         
           """) 
               
    rmse_df = pd.read_csv("streamlit/data/rmse_time_2020.csv")   
    
    MSE_r2 = go.Figure(data=[go.Table(
                header=dict(values=['RMSE','R^2 Score'],
                line_color='darkslategray',
                fill_color='#3366CC',
                align=['center','center'],
                font=dict(color='white', size=20),
                height=40),
                cells=dict(values=[rmse_df['RMSE'], rmse_df['R^2 Score']],
                line_color='darkslategray',
                fill=dict(color=['white', 'white']),
                align=['center', 'center'],
                font_size=18,
                height=30)
                  )])
    
    MSE_r2.update_layout(height = 300)
    st.plotly_chart(MSE_r2,use_container_width=True)           


    st.write("""
             The error by county can be plotted on a map as a percentage of the median sales price.  You can see below that most counties are below 2% error.
               """) 
               
    Predictions2020 = pd.read_csv("streamlit/data/Prediction_error_2020.csv")    
    
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)
        
    Pred2020_error = px.choropleth(Predictions2020, geojson=counties, locations='FIPS', color='Forecast error %',
                            color_continuous_scale="Viridis",
                            hover_name = 'County',
                            hover_data =['Predicted Median Sale Price 2020'],
                            scope="usa",
                            labels={'Forecast error %':'2020 % Forecast error'},
                            title = 'Average forecast error by ACS county for 2020 prediction, with most counties having less error than 2%'
                          )
    Pred2020_error.update_layout(height = 500)
    st.plotly_chart(Pred2020_error,use_container_width=True)


    st.header("Real Estate Predictions from VAR model  :clock8:")
    st.write("""
             We conclude the analysis by calculating the % increase of 2020 median sale price over 2019 median sale price and sorting for the top 10 counties that are predicted to have the highest % return.  
             
             
             """)

    summary2020prediction = pd.read_csv("streamlit/data/Summary_2020_Predictions.csv")    
    
    st.write(""" #           
             
              """)  
              
    forecasted_pct = px.choropleth(summary2020prediction, geojson=counties, locations='FIPS', color='Median Sale Price % increase',
                               color_continuous_scale="Viridis",
                                # range_color=(0, 100),
                               hover_name = 'County',
                               hover_data =['Predicted Median Sale Price 2020','Lower 95% Prediction Inverval','Upper 95% Prediction Inverval'],
                               scope="usa",
                               labels={'Mean_pred_price_pct':'% price change from 2019'},
                               title='All 613 ACS counties - 2020 average Median Sale Price % increase over 2019'
                              )
    forecasted_pct.update_layout(height = 500)    
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
    top_cnt.update_layout(height = 500)       
    st.plotly_chart(top_cnt,use_container_width=True)
    
    st.write(""" #           
             
              """)  

    st.write("This model predicts these 10 counties as having the highest percentage of growth in 2020 compared to 2019:")


    top10table = go.Figure(data=[go.Table(
                header=dict(values=['County','2020 Price % increase over 2019'],
                line_color='darkslategray',
                fill_color='#3366CC',
                align=['center','center'],
                font=dict(color='white', size=20),
                height=40,),
                cells=dict(values=[summary2020prediction[0:10]['County'], summary2020prediction[0:10]['Median Sale Price % increase']],
                line_color='darkslategray',
                fill=dict(color=['white', 'white']),
                align=['center', 'center'],
                font_size=18,
                height=30)
                  )])
    
    top10table.update_layout(width=1000, height=700)
    st.plotly_chart(top10table,use_container_width=True) 
    
    with st.expander("References"):
        st.write(
            """
        - https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VAR.html
        - https://www.qualtrics.com/support/stats-iq/analyses/regression-guides/interpreting-residual-plots-improve-regression/
        - https://stats.stackexchange.com/questions/55692/back-transformation-of-an-mlr-model
        """ 
        )
