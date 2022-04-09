"""Script designed to run the pytorch model for Team JIM Capstone."""
# standard imports
import streamlit as st
import pandas as pd
import numpy as np
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
# potentially can keep write up here, or make another file


def run_nn_model(period):
    """
    Run the pytorch forecasting model(s).
    """
    # st.write("Model 3 running...")

    # LOADING and DEFINING DATA
    if period == "Yearly":
        # load the yearly data
        st.write("Loading yearly data!")
        # TODO: Add the yearly data loading func from utils
        data["year"] = pd.to_datetime(data["year"], format="%Y")
        # set time index
        data["time_idx"] = data["year"].dt.year  # * 12 + data["year"].dt.month
        data["time_idx"] -= data["time_idx"].min()
        data["state_fips"] = data["state_fips"].astype("str").astype("category")
        data = None

        # define the dataset, i.e. add metadata to pandas dataframe for the model to understand it
        max_encoder_length = 4
        max_prediction_length = 1
        training_cutoff = (
            data["time_idx"].max() - max_prediction_length
        )  # day for cutoff

        training = None
        validation = None
        pass

    if period == "Monthly":
        # load the monthly data
        # TODO: Add the monthly data loading func from utils
        st.write("Loading monthly data!")
        data = None
        training = None
        validation = None
        pass

    validation = TimeSeriesDataSet.from_dataset(
        training,
        data,
        predict=True,
        min_prediction_idx=training.index.time.max() + 1,
        stop_randomization=True,
    )

    # convert datasets to dataloaders for training
    batch_size = 128
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=2
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size * 10,
        num_workers=2,  # double check factor of 10 will work
    )

    # RUNNING MODEL
    # baseline
    actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    baseline_predictions = Baseline().predict(val_dataloader)
    (actuals - baseline_predictions).abs().mean().item()

    st.success("Model sucessfully run!")

    # generate charts to display
