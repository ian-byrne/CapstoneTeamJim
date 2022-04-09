"""
Main module for display of the Capstone project for Team JIM.
"""
import streamlit as st

import warnings
from utils import load_data

# imports for training
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

# import dataset, network to train and metric to optimize
from pytorch_forecasting import (
    Baseline,
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    QuantileLoss,
    RMSE,
    MAE,
    MAPE,
    MASE,
    SMAPE,
)
from pytorch_forecasting.data import TimeSeriesDataSet, GroupNormalizer, NaNLabelEncoder
import torch

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

# the below imports are subject to change based on the final form of modules.
from model_one import model1
from model_two import model2
from model_three import run_nn_model


def main():
    """Display of the capstone project."""

    st.warning(
        "DEV - SUBJECT TO CHANGE"
    )  # TODO: remove before final production version.

    st.title("Capstone Project Team JIM")
    st.write("Authors: Jenny Ney, Melanie Starr, Ian Byrne")

    menu = ["Home", "Tree Analysis", "VAR Analysis", "RNN Analysis", "Conclusion"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("About page")
        st.write(
            """With this app the authors indend to the user to be able to follow along
            with design choices made in each model and view our results."""
        )
        with st.expander("Project Requirements"):
            st.write("""TODO: Add the requirements for the project here.""")

    if choice == "Tree Analysis":
        st.subheader("Tree Analysis")
        model1()

    if choice == "VAR Analysis":
        st.subheader("VAR Analysis")
        model2()

    if choice == "RNN Analysis":
        st.subheader("Neural Network Analysis")
        period = st.radio("Please select yearly or monthly data", ("Yearly", "Monthly"))
        # display text for post/analysis
        run_nn_model(period)

    if choice == "Conclusion":
        st.subheader("Conclusions")
        st.write("Summary of Capstone project conclusions.")


if __name__ == "__main__":
    main()
