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
from model_three import run_nn_model
from conclusion import conclusion


st.warning("DEV - SUBJECT TO CHANGE")  # TODO: remove before final production version.

st.title("Capstone Project Team JIM")
st.write("**Authors:** Jenny Ney, Melanie Starr, Ian Byrne")

menu = ["Home", "Supervised", "Time Series", "Neural Network", "Conclusion"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("About:")
    st.write(
        """With this app the authors indend to the user to be able to follow along
        with design choices made in each model and view our results."""
    )
    st.write(
        """
    Welcome to the **RE investment region analysis Capstone Project** for our Master of
    Applied Data Science from the University of Michigan School of Information!

    The authors Jenny Ney, Ian Byrne, and Melanie Starr formed their Capstone Team JIM
    with the intent of creating a framework to predict the direction of real estate
    markets after discussing a common interest in real estate investment.

    #### Data:

    In preparation for and during the early stages of the project, we identified open
    source data that would help us potentially make predictions on county home values.
    We ended up using the following resources in some form or another during the
    project: *insert data gathered*

    These data sources were originally stored in a Postgres database, however depending
    on when you are viewing this, that database may or may not be available. All data
    used within the final models of the project presented here can be found in the
    *github repo/drive/etc*

    #### Project:

    We each selected a model type that interested us and proved to test well on our
    preliminary data. This led to a Vector Autoregression (VAR) model, Tree model using
    the ExtraTreesRegressor, and *LSTM (confirm this)* based Transformer model being used in the project.
    We looked at several targets for each region, honing in on median sale price and the
    Home Price Index. As you will see, some models did better or worse depending on the
    target metric and the time granularity. In preparation for and in the early stages
    of the project, we identified open source data that would help us potentially
    make predictions on county home values. We ended up using the following resources
    in some form or another during the project: *insert data used*

    """
    )

    with st.expander("Project Requirements"):
        st.write(
            """
        - `pytorch-lightning==1.6.0`
        - `pytorch-forecasting==0.10.1`
        - `ADD OTHER PACKAGES IF NECESSARY`
        """  # TODO: Add other packages if needed
        )

if choice == "Supervised":
    st.header("Supervised Model")
    st.write("""Tree Analysis post.""")
    model1()

if choice == "Time Series":
    st.header("Time Series")
    st.write("""VAR Analysis post.""")
    model2()

if choice == "Neural Network":
    st.header("Neural Network")
    period = st.radio("Please select yearly or monthly data", ("Yearly", "Monthly"))
    # display text for post/analysis
    # run_nn_model(period)  # maybe add run button to reduce overhead

if choice == "Conclusion":
    st.header("Conclusion")
    st.write("Summary of Capstone project conclusions.")
    conclusion()