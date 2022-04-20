# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 09:43:45 2022

@author: melan
"""



import os
from datetime import datetime

import pandas as pd
import numpy as np
import pickle
from config import definitions
root_dir = definitions.root_directory()


from statsmodels.tsa.stattools import adfuller


from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults, VARResultsWrapper

from statsmodels.tsa.stattools import grangercausalitytests
from  sklearn.metrics import mean_squared_error,r2_score



import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
# render plot in default browser
pio.renderers.default = 'browser'


from urllib.request import urlopen
import json

pd.set_option('display.max_columns',None)

now = datetime.now()
now = now.strftime("%d%b%Y_%Hh%M")


def time_series():

    # read in the processed data
    df=pd.read_pickle(os.path.join(root_dir,"CapstoneTeamJim","data","processed","data_log_2016_2021_VARcountysubset.pkl"))
    
    # pivot the data
    county_name = df[['county_fips','region']].drop_duplicates()
    df_pivot = df.pivot(index='date',columns='county_fips',values='log_median_sale_price')
    
    # continue to forecast 2021 to extract error for 2022 correction and for confidence intervals
    df_time = df[((df['year']>='2016')&(df['year']<'2021'))]
    df_time_log = df_time.pivot(index='date',columns='county_fips',values='log_median_sale_price')
    

    # def adfuller_func(dataframe):
    #     res = []
    #     for col in list(dataframe.columns):
    #         r = adfuller(dataframe[col], autolag='AIC')
    #         res.append(round(r[1],3))
    #     return res
    
    
    # check stationarity with ad fuller testing - no differencing
    # no_diff = adfuller_func(df_time_log)
    # adf = pd.DataFrame(data=[no_diff],columns=list(df_time_log.columns))
    # adf = adf.melt()
    # adf.columns=['county_fips','no_diff_p']
    # adf[adf['no_diff_p']<0.05].count()
    # 451/1393 are stationary
    
    # check stationarity with ad fuller testing - first differencing
    df_time_diff_log = df_time_log.diff().dropna()
    # first_diff = adfuller_func(df_time_diff_log)
    # adf['first_diff_p'] = first_diff 
    # adf[adf['first_diff_p']<0.05].count()
    # 1348/1393 are stationary
    
    df_time_2diff_log = df_time_diff_log.diff().dropna()
    # sec_diff = adfuller_func(df_time_2diff_log)
    # adf['sec_diff_p'] = sec_diff 
    # adf[adf['sec_diff_p']<0.05].count()
    # 1390/1393 are stationary
    
    
    df_diff = df_pivot.diff().dropna()
    df_2diff = df_diff.diff().dropna()
    
    #### FORECAST 2021 TO EXTRACT ERROR FOR 2022 CORRECTION
    p=12
    
    num_forecasts = 12
    var_res2021, forecasts = None, None
    
    
    # using second differencing
    model2021 = VAR(df_time_2diff_log,freq='MS')
    
    # fit 2021 model
    var_res2021 = model2021.fit(maxlags=p)

    # forecast 2021 values 
    forecasts = var_res2021.forecast(df_time_2diff_log.values,steps=num_forecasts)
    
    # set index of forecast df
    dfindex = pd.date_range(start=df_time_log.index[-1],periods = num_forecasts+1, freq='MS')[1:]
    
    # first undifferencing
    lastvals_second_diff = df_time_log.diff()[-1:]
    for_df = pd.DataFrame(data = forecasts,columns= df_time_log.columns, index = dfindex)
    for_df = pd.concat([lastvals_second_diff, for_df],axis=0,ignore_index=False).cumsum()[1:]
    # second undifferencing
    lastvals_first_diff = df_time_log[-1:]
    for_df = pd.concat([lastvals_first_diff, for_df],axis=0,ignore_index=False).cumsum()[1:]
    
    
    #### FORECAST 2022
    
    p=12
    
    num_forecasts = 12
    var_res, forecasts = None, None
    
    
    # using second differencing
    model = VAR(df_2diff,freq='MS')
    
    # fit model    
    var_res = model.fit(maxlags=p)

    # forecast 2022 values
    forecasts = var_res.forecast(df_2diff.values,steps=num_forecasts)
    
    # set index of forecast df
    dfindex = pd.date_range(start=df_pivot.index[-1],periods = num_forecasts+1, freq='MS')[1:]
    
    # first undifferencing    
    lastvals_second_diff = df_pivot.diff()[-1:]
    for_df2022 = pd.DataFrame(data = forecasts,columns= df_pivot.columns, index = dfindex)
    for_df2022 = pd.concat([lastvals_second_diff, for_df2022],axis=0,ignore_index=False).cumsum()[1:]
    # second undifferencing    
    lastvals_first_diff = df_pivot[-1:]
    for_df2022 = pd.concat([lastvals_first_diff, for_df2022],axis=0,ignore_index=False).cumsum()[1:]
    
     
    ###  TESTING USING HOLDOUT 2021 DATA
    df2021 = df[df['year']=='2021']
    
    
    pred = pd.melt(for_df.reset_index(),id_vars='index',value_name = 'Pred_log_median_sale_price')
    pred.columns=['date','county_fips','Pred_log_median_sale_price']
    pred = pred.merge(df2021, on=['county_fips','date'])
    # calculate the log_errors
    pred['log_pred_errors'] = pred['log_median_sale_price']-pred['Pred_log_median_sale_price']
    hist = px.histogram(pred['log_pred_errors'],title='Histogram of residual error of model - appears normally distributed')
    
    hist.write_image(os.path.join(root_dir,"CapstoneTeamJim","reports","figures","Time_Series","Hist_residual_error_model_testing.png"),width=1980, height=1080)
    
    
    ###  Plot residual errors  (still in log transform)
    # https://www.qualtrics.com/support/stats-iq/analyses/regression-guides/interpreting-residual-plots-improve-regression/
    ## Keep - shows that errors 
    # (1) they’re pretty symmetrically distributed, tending to cluster towards the middle of the plot.
    # (2) they’re clustered around the lower single digits of the y-axis (e.g., 0.5 or 1.5, not 30 or 150).
    # (3) in general, there aren’t any clear patterns.
    pred_errors = px.scatter(pred,x='log_median_sale_price',y='log_pred_errors', title='Residual error clustered around middle of plot with tight range and no clear patterns')
    pred_errors.write_image(os.path.join(root_dir,"CapstoneTeamJim","reports","figures","Time_Series","Scatter_residual_error_model_testing.png"),width=1980, height=1080)
    
    ###   Transform log predictions to original units
    
    # https://stats.stackexchange.com/questions/55692/back-transformation-of-an-mlr-model
    
    # calculate the mean transformed error by county
    pred['pred_errors_trans'] = np.exp(pred['log_pred_errors'])
    mean_transformed_error = pred.groupby('county_fips')['pred_errors_trans'].mean().reset_index(name='Mean_pred_error_trans')
    
    # transform predicted median sale price back to dollars
    pred['pred_med_sale_trans_biased'] = np.exp(pred['Pred_log_median_sale_price'])
    pred = pred.merge(mean_transformed_error,on='county_fips')
    pred['pred_med_sale_trans']  = pred['pred_med_sale_trans_biased']*pred['Mean_pred_error_trans']
    
    
    
    # Prediction by county for 2021
    county_mean_pred2021 = pred.groupby('county_fips')['pred_med_sale_trans'].mean().reset_index(name='Predicted_mean_2021')
    county_mean = pred.groupby('county_fips')['median_sale_price'].mean().reset_index(name='County_mean')
    Predictions2021 = county_mean_pred2021.merge(county_mean,on='county_fips')
    # add county name for map
    Predictions2021 = Predictions2021.merge(county_name,on='county_fips')
    Predictions2021['error'] = Predictions2021['County_mean']-Predictions2021['Predicted_mean_2021']
    Predictions2021['error_pct'] = np.absolute(Predictions2021['error'])/Predictions2021['County_mean']*100
    
   
    
    # All 1393 counties
    rmse_calc = mean_squared_error(Predictions2021['County_mean'], Predictions2021['Predicted_mean_2021'],squared=False) 
    rmse_calc = ("${:,.0f}".format(rmse_calc))
    # $2407.2233765332826
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    r2_calc = r2_score(Predictions2021['County_mean'], Predictions2021['Predicted_mean_2021']).round(5) 
    # 0.9998014759535951
    
    rmse_df = pd.DataFrame(data = [[rmse_calc,r2_calc]], columns = ['RMSE','R^2 Score'])
    rmse_df.to_csv(os.path.join(root_dir,"CapstoneTeamJim","reports","results","Time_Series","rmse_time_2022.csv"),index=False)
    
    
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
        
    MSE_r2.update_layout(title_text = "Model error for 2022 Predictions", width=500, height=300)
    MSE_r2.show()
    MSE_r2.write_image(os.path.join(root_dir,"CapstoneTeamJim","reports","figures","Time_Series","Table_Time_series_model_error_2022_predictions.png"),width=500, height=300)
    
   
    
    ###  PREDICTIONS USING ALL DATA

    pred2022 = pd.concat([df_pivot, for_df2022],axis=0,ignore_index=False)
    pred_long = pd.melt(pred2022.reset_index(),id_vars='index',value_name = 'Pred_log_median_sale_price')
    pred_long.columns = ['date','county_fips','log_median_sale_price']
    pred_long['year'] = pred_long['date'].dt.year
    # merge 2021 transformed error onto df
    pred_long = pred_long.merge(mean_transformed_error,on=['county_fips'] )
    # set error for years previous to 2022 = 1
    pred_long['Mean_pred_error_trans'][pred_long['year']!=2022]=1
    # Transform log median price to orignal dollars
    pred_long['Med_price_trans_biased'] = np.exp(pred_long['log_median_sale_price'])
    # multiply median price by error term
    pred_long['Mean_pred_price_trans'] = pred_long['Med_price_trans_biased'] * pred_long['Mean_pred_error_trans']
    
    prediction_std = pred_long[pred_long['year']==2022].groupby('county_fips')['Mean_pred_price_trans'].std().reset_index(name='Predicted_med_price_std')
    
    
    # group by county and year to get average median sale price 
    Predictions_by_county = pred_long.groupby(['county_fips','year'])['Mean_pred_price_trans'].mean().reset_index(name='Mean_median_sale_price')
    # calculate the difference from 2022 to 2021
    Predictions_by_county['Mean_pred_price_diff'] = Predictions_by_county.groupby('county_fips')['Mean_median_sale_price'].transform(lambda v: v.diff()).shift(-1)
    # add county name for map
    Predictions_by_county = Predictions_by_county.merge(county_name,on='county_fips')
    # calculation the % change from 2021
    Predictions_by_county['Mean_pred_price_pct'] = Predictions_by_county['Mean_pred_price_diff']/Predictions_by_county['Mean_median_sale_price']*100
    
    Predictions_by_county_uncorrected = pred_long.groupby(['county_fips','year'])['Med_price_trans_biased'].mean().reset_index(name='Mean_median_sale_price')
    # calculate the difference from 2022 to 2021
    Predictions_by_county_uncorrected['Mean_pred_price_diff'] = Predictions_by_county_uncorrected.groupby('county_fips')['Mean_median_sale_price'].transform(lambda v: v.diff()).shift(-1)
    # add county name for map
    Predictions_by_county_uncorrected = Predictions_by_county_uncorrected.merge(county_name,on='county_fips')
    # calculation the % change from 2021
    Predictions_by_county_uncorrected['Mean_pred_price_pct'] = Predictions_by_county_uncorrected['Mean_pred_price_diff']/Predictions_by_county_uncorrected['Mean_median_sale_price']*100
    Top_counties_uncorrected = Predictions_by_county_uncorrected[Predictions_by_county_uncorrected['year']==2021].sort_values(by=['Mean_pred_price_pct'],ascending=False)
    
    
    Predictions2022 = Predictions_by_county_uncorrected[Predictions_by_county_uncorrected['year']==2022][['county_fips','region','Mean_median_sale_price']]
    Predictions2022.columns = ['county_fips','region','Pred_mean_med_sale_price_2022']
    summary2022prediction = Top_counties_uncorrected.merge(Predictions2022,how='left',on=['county_fips','region']).drop(columns=['year'])
    # summary2022prediction = summary2022prediction.merge(Predictions2021,how='left',on=['county_fips','region']).drop(columns=['County_mean'])
    summary2022prediction.columns = ['county_fips','Mean_med_sale_price_2021','Mean_pred_price_yoy','county','Mean_pred_price_pct_yoy','Pred_mean_med_sale_price_2022']
    
    # Calculating confidence intervals and formatting for charts
    summary2022prediction = summary2022prediction.merge(prediction_std,how='left',on=['county_fips'])
    summary2022prediction['lower_95'] = summary2022prediction['Pred_mean_med_sale_price_2022']-(summary2022prediction['Predicted_med_price_std']*1.96)
    summary2022prediction['upper_95'] = summary2022prediction['Pred_mean_med_sale_price_2022']+(summary2022prediction['Predicted_med_price_std']*1.96)
    summary2022prediction = summary2022prediction[['county_fips','county','Mean_med_sale_price_2021','Pred_mean_med_sale_price_2022','lower_95','upper_95','Mean_pred_price_yoy','Mean_pred_price_pct_yoy']]
    summary2022prediction['Mean_med_sale_price_2021'] = summary2022prediction['Mean_med_sale_price_2021'].map("${:,.0f}".format)
    summary2022prediction['Pred_mean_med_sale_price_2022'] = summary2022prediction['Pred_mean_med_sale_price_2022'].map("${:,.0f}".format)
    summary2022prediction['lower_95'] = summary2022prediction['lower_95'].map("${:,.0f}".format)
    summary2022prediction['upper_95'] = summary2022prediction['upper_95'].map("${:,.0f}".format)
    summary2022prediction['Mean_pred_price_yoy'] = summary2022prediction['Mean_pred_price_yoy'].map("${:,.0f}".format)
    summary2022prediction['Mean_pred_price_pct_yoy'] = summary2022prediction['Mean_pred_price_pct_yoy'].round(2)
    summary2022prediction = summary2022prediction.sort_values(by='Mean_pred_price_pct_yoy',ascending=False)
    
    summary2022prediction.columns = ['FIPS','County','Median Sale Price 2021','Predicted Median Sale Price 2022','Lower 95% Prediction Inverval','Upper 95% Prediction Inverval','Median Sale Price increase','Median Sale Price % increase',]
    
    summary2022prediction.to_csv(os.path.join(root_dir,"CapstoneTeamJim","reports","results","Time_Series","Summary_2022_Predictions.csv"),index=False)
    
    
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
    Pred2022_plot.write_image(os.path.join(root_dir,"CapstoneTeamJim","reports","figures","Time_Series","Choro_Summary_all_counties2022_predictions.png"),width=1980, height=1080)
    
    
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
    Top_10_Pred2022_plot.write_image(os.path.join(root_dir,"CapstoneTeamJim","reports","figures","Time_Series","Choro_Summary_top10_counties_2022_predictions.png"),width=1980, height=1080)

if __name__ == "__main__":
    time_series()


























