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
import os

# standard imports
import numpy as np
import pandas as pd
import argparse
from config import definitions

root_dir = definitions.root_directory()

# TODO: ensure the correct paths are being used here once the final model is done.
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
            print("Loading yearly data...")
            path = os.path.join(
                root_dir, "CapstoneTeamJim", "data", "processed", "fulldataset.pkl"
            )
            data = pd.read_pickle(path)
            data["year"] = pd.to_datetime(data["year"], format="%Y")

            # columns to drop
            drop_cols = [
                "travel_time_to_work",
                "11_avg_annual_employee_pct_chg",
                "21_avg_annual_employee_pct_chg",
                "22_avg_annual_employee_pct_chg",
                "23_avg_annual_employee_pct_chg",
                "42_avg_annual_employee_pct_chg",
                "51_avg_annual_employee_pct_chg",
                "52_avg_annual_employee_pct_chg",
                "53_avg_annual_employee_pct_chg",
                "54_avg_annual_employee_pct_chg",
                "55_avg_annual_employee_pct_chg",
                "56_avg_annual_employee_pct_chg",
                "61_avg_annual_employee_pct_chg",
                "62_avg_annual_employee_pct_chg",
                "71_avg_annual_employee_pct_chg",
                "72_avg_annual_employee_pct_chg",
                "81_avg_annual_employee_pct_chg",
                "92_avg_annual_employee_pct_chg",
                "99_avg_annual_employee_pct_chg",
                "11_avg_annual_pay_pct_chg",
                "21_avg_annual_pay_pct_chg",
                "22_avg_annual_pay_pct_chg",
                "23_avg_annual_pay_pct_chg",
                "42_avg_annual_pay_pct_chg",
                "51_avg_annual_pay_pct_chg",
                "52_avg_annual_pay_pct_chg",
                "53_avg_annual_pay_pct_chg",
                "54_avg_annual_pay_pct_chg",
                "55_avg_annual_pay_pct_chg",
                "56_avg_annual_pay_pct_chg",
                "61_avg_annual_pay_pct_chg",
                "62_avg_annual_pay_pct_chg",
                "71_avg_annual_pay_pct_chg",
                "72_avg_annual_pay_pct_chg",
                "81_avg_annual_pay_pct_chg",
                "92_avg_annual_pay_pct_chg",
                "99_avg_annual_pay_pct_chg",
                "vehicles_per_person",
                "population",
                "household_income",
                "home_value_median",
                "birth_15_19_pct",
                "birth_20_24_pct",
                "birth_25_29_pct",
                "birth_30_34_pct",
                "birth_35_39_pct",
                "birth_40_44_pct",
                "birth_45_50_pct",
                "grade12_nodiploma_pct",
                "hs_diploma_pct",
                "some_college_lessthan_1yr_pct",
                "some_college_greaterthan_1yr_pct",
                "bachelor_degree_pct",
                "master_degree_pct",
                "professional_degree_pct",
                "doctorate_degree_pct",
                "occupied_units_pct",
                "vacant_units_pct",
                "median_ppsf",
                "annual_change_pct",
            ]
            data = data.drop(columns=drop_cols)
            data = data.fillna(0)
            data["time_idx"] = data["year"].dt.year  # * 12 + data["year"].dt.month
            data["time_idx"] -= data["time_idx"].min()
            data["state_fips"] = data["state_fips"].astype("str").astype("category")
        except:
            print("Check pathing.")

    elif granularity == "month":
        try:
            # NOTE: this file will not be in the repo, compressed it is 100+mb
            print("Loading monthly data...")
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

    return data


# TODO: Adjust functionality if I don't get to all the features.
def train_model(granularity, data):
    """
    Train the pytorch-forecasting TemporalFusionTransformer.

    Parameters
    ----------
    granularity: str
        Granularity for the mondel data. Can be year or month.
    data: pandas DataFrame
        DataFrame that will house the training and validation data.
    Returns
    -------

    """
    if granularity == "year":
        # define the dataset, i.e. add metadata to pandas dataframe for the model to understand it
        max_encoder_length = 3
        max_prediction_length = 1
        training_cutoff = (
            data["time_idx"].max() - max_prediction_length
        )  # day for cutoff

        training = TimeSeriesDataSet(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",  # column name of time of observation
            target="median_sale_price",  # column name of target to predict
            group_ids=[
                "state_fips",
                "county_fips",
            ],  # column name(s) for timeseries IDs
            max_encoder_length=max_encoder_length,  # how much history to use
            max_prediction_length=max_prediction_length,  # how far to predict into future
            # covariates static for a timeseries ID
            static_categoricals=["state_fips", "county_fips"],
            static_reals=[],
            # covariates known and unknown in the future to inform prediction
            time_varying_known_categoricals=[],
            time_varying_known_reals=[
                "income_tax_low",
                "income_tax_high",
                "corp_income_tax_low",
                "corp_income_tax_high",
            ],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                "debt_ratio_low",
                "debt_ratio_high",
                "median_sale_price",
                "median_list_price",
                "median_list_ppsf",
                "homes_sold",
                "new_listings",
                "inventory",
                "months_of_supply",
                "median_dom",
                "avg_sale_to_list",
                "sold_above_list",
                "gdp",
            ],
            add_relative_time_idx=True,
            allow_missing_timesteps=True,
            categorical_encoders={"county_fips": NaNLabelEncoder(add_nan=True)},
            add_target_scales=True,
            add_encoder_length=True,
        )

    if granularity == "month":
        max_encoder_length = 60
        max_prediction_length = 12
        training_cutoff = (
            data["time_idx"].max() - max_prediction_length
        )  # day for cutoff

        training = TimeSeriesDataSet(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",  # column name of time of observation
            target="median_sale_price",  # column name of target to predict
            group_ids=[
                "county_fips",
                "region",
                "state",
                "property_type",
            ],  # column name(s) for timeseries IDs
            max_encoder_length=max_encoder_length,  # how much history to use
            max_prediction_length=max_prediction_length,  # how far to predict into future
            # covariates static for a timeseries ID
            static_categoricals=["region", "property_type", "state", "county_fips"],
            static_reals=[],
            # covariates known and unknown in the future to inform prediction
            time_varying_known_categoricals=["month"],
            time_varying_known_reals=[
                "year",
                "time_idx",
                "income_tax_low",
                "income_tax_high",
                "corp_income_tax_low",
                "corp_income_tax_high",
            ],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                "median_sale_price",
                "median_list_price",
                "median_list_ppsf",
                "homes_sold",
                "pending_sales",
                "new_listings",
                "inventory",
                "months_of_supply",
                "median_dom",
                "sold_above_list",
                "off_market_in_two_weeks",
            ],
            add_relative_time_idx=True,
            allow_missing_timesteps=True,
            categorical_encoders={
                "region": NaNLabelEncoder(add_nan=True),
                "county_fips": NaNLabelEncoder(add_nan=True),
            },
        )
    # create validation dataset using the same normalization techniques as for the training dataset
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

    # configure network and trainer
    # create PyTorch Lightning Trainer with early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=4, verbose=False, mode="min"
    )
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

    trainer = pl.Trainer(
        max_epochs=30,
        gpus=0,  # run on CPU, if on multiple GPUs, use accelerator="ddp"
        gradient_clip_val=0.15,
        limit_train_batches=30,  # 30 batches per epoch
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    if granularity == "monthly":
        tft = TemporalFusionTransformer.from_dataset(
            # dataset
            training,
            # architecture hyperparameters
            hidden_size=58,
            attention_head_size=4,  # play around with this param - can go up to 4 depending on sizeof data
            dropout=0.1,
            hidden_continuous_size=56,
            # loss metric to optimize
            loss=QuantileLoss(),
            # logging frequency
            log_interval=0,
            # optimizer parameters
            learning_rate=0.007,  # change back to best_tft
            # reduce learning rate if no improvement in validation loss after x epochs
            reduce_on_plateau_patience=4,
        )
    else:
        tft = TemporalFusionTransformer.from_dataset(
            # dataset
            training,
            # architecture hyperparameters
            hidden_size=82,
            lstm_layers=1,  # Check to see if this helps, default is 1 - 2 didnt seem to to help
            attention_head_size=1,  # play around with this param - can go up to 4 depending on sizeof data
            dropout=0.15,
            hidden_continuous_size=22,
            # loss metric to optimize
            loss=QuantileLoss(),
            # logging frequency
            log_interval=0,
            # optimizer parameters
            learning_rate=0.1,  # change back to best_tft
            # reduce learning rate if no improvement in validation loss after x epochs
            reduce_on_plateau_patience=4,
        )

    # fit the model on the data - redefine the model with the correct learning rate if necessary
    trainer.fit(
        tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    return best_tft


def main(granularity, pred_period):
    """Run the model with desired parameters."""
    data = load_data(granularity)
    train_model(granularity, data)
    print("TFT Model trained for 1 year predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs the TemporalFusionTransformer model."
    )
    parser.add_argument(
        "--granularity",
        nargs="?",
        default="year",
        help="Time granularity for the model. Choose year or month, defaults to year.",
    )

    args = parser.parse_args()

    granularity = args.granularity

    print("Running model now...")
    main(granularity)
    print("Training completed.")
