"""Example Model file."""
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from urllib.request import urlopen
import json

import plotly.graph_objects as go
import plotly.express as px


def model1():

    st.header("Extra Trees Regressor  :deciduous_tree: 	")
    
    st.write("We wanted to explore tree regression models because of their capabilities with diverse input data. This family of models handles the variety of scale in our data well. We were able to utilize all data sources including demographic data from the ACS. ")

    st.write("The use of ACS data limits the counties available to those with populations > 65,000. In order to compare to the VAR model (model 2) which uses monthly data, we are also limited to counties with monthly Redfin data. The final count of U.S. counties used in this model is 613.")
    
    st.write("Using PyCaret AutoML we isolated two top-performing model types on our data: Random Forest Regressor and Extra Trees Regressor. We then further tuned these models outside of PyCaret through gridsearch. Ultimately the Extra Trees Regressor performed best and we selected it as our top tree model.")

    st.write("We split our data into train, test, and validation. Because the data is yearly we withheld all of 2019 (predicting 2020 median sale price) as validation. Training data comprised 75% of the remaining years with the remainder to test.")

    st.write("Our first iteration of this model yielded extremely strong results with a test R2 of 0.973 and validation R2 of 0.951. Such high scores were somewhat suspect and further explained by examining the features importances:")

    feature_df1 = pd.read_csv("capstoneteamjim/main/streamlit/data/tree_model1_features.csv")

    fig = px.bar_polar(feature_df1.iloc[:30,:], r='Importance', theta='Feature',
                color='Feature', template='plotly_dark',
                color_discrete_sequence=px.colors.sequential.Plasma_r)

    st.plotly_chart(fig,use_container_width=True)

    st.write("The top four features in the model are median home value, median price per square foot, median list price per square foot, and median list price. We suspected high correlation with the outcome variable, median sale price, and confirmed via correlation matrix:")

    corr_df = pd.read_csv("capstoneteamjim/main/streamlit/data/correlation_matrix.csv").drop("Unnamed: 0", axis=1)
    corr_df.columns = ['median home value', 'median price per square foot', 'median list price', 'median list price per square foot', 'median sale price']
    mask = np.triu(np.ones_like(corr_df.corr(), dtype=np.bool))
    fig, ax = plt.subplots()
    sns.heatmap(corr_df.corr(), mask=mask, annot=True, cmap='Blues', ax=ax)

    st.pyplot(fig, use_container_width=True)

    st.write("For our final iteration of this model type we removed the four highly correlated features and ran the model with the remaining features. We still found highly favorable results with a test R2 of 0.92 and 2019 validation data R2 of 0.829. The RMSE (a measure of the standard deviation of residuals) for 2019 predicting 2020 with this model is $66,606.")

    st.write("The feature importances show much different results in this iteration:")

    feature_df2 = pd.read_csv("capstoneteamjim/main/streamlit/data/tree_model2_features.csv")

    fig = px.bar_polar(feature_df2.iloc[:30,:], r='Importance', theta='Feature',
                color='Feature', template='plotly_dark',
                color_discrete_sequence=px.colors.sequential.Plasma_r)

    st.plotly_chart(fig,use_container_width=True)

    st.write("Some of the top features include tax and debt data. These are at the state level. Some of the county-level demographic data also shows up here. Education levels in particular can predict higher home sales. Average sale to list is a measure of time on the market.")

    st.write("A plot of counties along with the predicted and real values in 2020 is shown below:")

    data2019 = pd.read_csv("capstoneteamjim/main/streamlit/data/prediction_map.csv")

    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    fig = px.choropleth(data2019, geojson=counties, locations='FIPS', color='2020 % Forecast Error',
                            color_continuous_scale="Viridis",
                            hover_name = 'County',
                            hover_data =['Predicted Median Sale Price 2020'],
                            scope="usa",
                            labels={'Forecast error %':'2020 % Forecast error'},
                            title = 'Average forecast error by ACS county for 2020 prediction'
                            )

    st.plotly_chart(fig,use_container_width=False)

    st.write("Metrics for this model are:")

    rmse_df = pd.DataFrame({'RMSE': ['$66,606'], 'Test R^2': ['0.92'], 'Validation R^2': ['0.829']})
    
    MSE_r2 = go.Figure(data=[go.Table(
                header=dict(values=['RMSE','Test R^2', 'Validation R^2'],
                line_color='darkslategray',
                fill_color='#3366CC',
                align=['center','center', 'center'],
                font=dict(color='white', size=20),
                height=40),
                cells=dict(values=[rmse_df['RMSE'], rmse_df['Test R^2'], rmse_df['Validation R^2']],
                line_color='darkslategray',
                fill=dict(color=['white', 'white', 'white']),
                align=['center', 'center', 'center'],
                font_size=18,
                height=30)
                  )])
    
    MSE_r2.update_layout(height = 300)
    st.plotly_chart(MSE_r2,use_container_width=True) 

    st.header("Top 10 Counties from Extra Trees Regressor model  :evergreen_tree:")
    st.write("Below are the top 10 predicted counties for model comparison")


    summary2020prediction = pd.DataFrame({'County': ['Athens County, OH', 'Gloucester County, NJ', 'Camden County, NJ',
                                    'Burlington County, NJ', 'Mercer County, NJ', 'Sussex County, NJ', 'Cumberland County, NJ',
                                    'Coryell County, TX'], '2020 Price % increase over 2019': [143,123,99,85,74,71,65,59]})



    top10table = go.Figure(data=[go.Table(
                header=dict(values=['County','2020 Price % increase over 2019'],
                line_color='darkslategray',
                fill_color='#3366CC',
                align=['center','center'],
                font=dict(color='white', size=20),
                height=40,),
                cells=dict(values=[summary2020prediction['County'], summary2020prediction['2020 Price % increase over 2019']],
                line_color='darkslategray',
                fill=dict(color=['white', 'white']),
                align=['center', 'center'],
                font_size=18,
                height=30)
                  )])
    
    top10table.update_layout(width=1000, height=500)
    st.plotly_chart(top10table,use_container_width=True)


    st.write("""Some drawbacks of this model are: 
             
*  Time limitations. Because demographic data is only available from 2015-2019 at the county level we cannot make longer predictions or predict into the future.
*  Granularity: county level is the smallest available location type available across this dataset.
*  Lack of monthly data availability, unlike the other two models. This is due to data availability of ACS demographic data.
*  Feature importances are all positive, meaning we cannot discern whether a feature has a negative or positive correlation with the outcome variable.
                 
              """) 

    st.write("""Future work:
             
*  As years go by the availability of demographic data will increase. This model clearly shows the importance of some of these features. Even if we could not gather current demographic data, with a longer history we could build a regression model to predict more than 1 year ahead, making predictions into the future possible. This would be a useful tool for investors and individuals in the market for a home.
*  Incorporate ensemble models to boost prediction strength.

                 
              """) 




    # Insert plotly plots

    # st.plotly_chart(Pred2022_plot,use_container_width=True)




    # Insert plotly plots

    # st.plotly_chart(Pred2022_plot,use_container_width=True)