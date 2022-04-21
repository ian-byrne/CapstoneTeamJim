"""
Main module for display of the Capstone project for Team JIM.
"""
import streamlit as st
import warnings
from utils import load_data


warnings.filterwarnings("ignore")  # avoid printing out absolute paths

# the below imports are subject to change based on the final form of modules.
from model_one import model1
from model_two import model2
from model_three import pytorch_writeup
from conclusion import conclusion


st.warning(
    "This analysis is for a Capstone project and the result have not been vetted by a licensed professional.  Please invest responsibly."
)  # TODO: remove before final production version.

st.title("Capstone Project Team JIM")
st.write("**Authors:** Jenny Ney, Ian Byrne, Melanie Starr")

menu = ["Home", "Supervised", "Time Series", "Neural Network", "Conclusion"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("About:")
    st.write(
        """With this app the authors intend for the user to be able to follow along
        with design choices made in each model and view our results.
    """
    )
    st.write(
        """
    Welcome to the **Real Estate Investment Region Analysis Capstone Project** for our Master of
    Applied Data Science from the University of Michigan School of Information!
               """
    )
    st.write(
        """
    The authors Jenny Ney, Ian Byrne, and Melanie Starr formed their Capstone Team JIM
    with the intent of creating a framework to predict the direction of real estate
    markets after discussing a common interest in real estate investment.
        """
    )
    st.write(
        """
    #### Data:

    In preparation for and during the early stages of the project, we identified open
    source data that would help us potentially make predictions on county home values.
    The following resources were used, in some form or another, during the
    project:
                """
    )
    st.write(
        """
*  Bureau of Labor Statistics Wage Data
*  American Community Survey Demographic Data
*  County Debt Ratio
*  State Personal and Corporate Income Tax Rates
*  Redfin Real Estate Data
*  Unemployment Rates
*  World Bank GDP Data
        """
    )

    st.write(
        """
    These data sources were originally stored in a Postgres database, however depending
    on when you are viewing this, that database may or may not be available.  All data
    used within the final models of the project presented here can be found in the
    github repository https://github.com/ian-byrne/CapstoneTeamJim
        """
    )
    st.write(
        """
    #### Project:

    We each selected a model type that interested us and proved to test well on our
    preliminary data. This led to a time series Vector Autoregression (VAR) model, a supervised ExtraTreesRegressor model, and a neural network Transformer model being used in the project.  We looked at several targets for each region, and concluded that median sale price would be our final prediction target as it's more intuitive for the end user to understand.  As you will see, some models did better or worse depending on the data inputs and time granularity.

    """
    )

    with st.expander("Project Requirements"):
        st.write(
            """
        - `matplotlib==3.5.1`
        - `pandas>=1.4.1,<=1.4.2`
        - `pytorch-lightning==1.6.0`
        - `pytorch-forecasting==0.10.1`
        - `plotly>=5.6.0,<=5.7.0`
        - `seaborn==0.11.2`
        - `sklearn==0.0`
        """  
        )

if choice == "Supervised":
    st.header("Supervised Model")
    st.write(
        """Supervised learning allows for custom model fit. Tree models in particular are able to handle a variety of data types. This model is the only one able to incorporate demographic data from ACS."""
    )
    model1()

if choice == "Time Series":
    st.header("Time Series")
    st.write(
        """Time Series modeling looks for patterns in time sequenced datasets in order to predict the future.  The time series model selected for this project is the Vector Autoregression model."""
    )
    model2()

if choice == "Neural Network":
    st.header("Neural Network")
    st.write(
        """
    Modeled using the Temporal Fusion Transformer architecture. This is a form of
    deep neural network designed with an attention mechanism to allow it to refer
    back to long term dependencies."""
    )
    # display text for post/analysis
    pytorch_writeup()

if choice == "Conclusion":
    st.header("Conclusion")
    st.write(
        "Based on a comparison on the metrics between the models, the time series model performed the best."
    )
    conclusion()
