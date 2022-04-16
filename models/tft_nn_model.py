"""TemporalFusionTransformer based model for UMich MADS Capstone."""

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore")

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
)
from pytorch_forecasting.data import TimeSeriesDataSet, GroupNormalizer, NaNLabelEncoder
from sklearn.metrics import mean_squared_error
import torch

# standard imports
import numpy as np
import pandas as pd

# ensure the correct paths are being used here once the final model is done.
def load_data(granularity="year"):
    """
    Load data for the pytorch-forecasting model.

    Parameters
    ----------
    granularity: str
        Granularity for the model data. Choose year or month.

    Returns
    --------
    data: pandas DataFrame
        Pandas dataframe containing processed data for NN model.
    granularity: str
        Same granularity variable from above to be passed to NN model.

    """
    if granularity == "year":
        try:
            data = pd.read_pickle("data/fulldataset.pkl")
            data["year"] = pd.to_datetime(data["year"], format="%Y")
        except:
            print("Check pathing.")

    elif granularity == "month":
        try:
            data = pd.read_csv(
                "nn_monthly_data.csv.gz", parse_dates=["period_begin", "period_end"]
            )
            # columns to drop
            cols = [
                "period_end",
                "region_type",
                "region_type_id",
                "is_seasonally_adjusted",
                "property_type_id",  # leave property type
                "state_x",
                "fips",
                "state_y",
                "state_fips",
                "table_id",
                "period_duration",
                "city",
                "state_code",
                "parent_metro_region",
                "parent_metro_region_metro_code",
                "median_ppsf",
                "avg_sale_to_list",
                "last_updated",
                "county",
            ]
            data["time_idx"] = (
                data["period_end"].dt.year * 12 + data["period_end"].dt.month
            )
            data["time_idx"] -= data["time_idx"].min()
            data["month"] = data["period_end"].dt.month.astype(str).astype("category")

            data.drop(columns=cols, inplace=True)
            # drop the yoy and mom columns
            data.drop(list(data.filter(regex="yoy|mom")), axis=1, inplace=True)
            data["county_fips"] = data["county_fips"].astype(str).astype("category")
        except:
            print(
                """Please contact authors for monthly data.
                This dataset was too large to be stored in github."""
            )
    else:
        print("Please select from year or month granularity.")

    return data, granularity


def train_model(granularity, data):
    """
    Train the pytorch-forecasting TemporalFusionTransformer.

    Parameters
    ----------
    granularity: str

    data: pandas DataFrame

    Returns
    -------

    """
    if granularity == "year":
        pass

    if granularity == "month":
        pass

    return None
