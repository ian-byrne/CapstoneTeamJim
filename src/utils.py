"""Utility functions for capstone project. Includes data loading functions."""

import numpy as np
import pandas as pd


def load_data(year):
    """Load the yearly data from pickle file in the data directory."""
    if year == 2018:
        return pd.read_pickle("data/dataraw_to2018.pkl")
    if year == 2019:
        return pd.read_pickle("data/dataraw_2019.pkl")
    else:
        print("No specified year, loading through 2018")
        return pd.read_pickle("data/dataraw_to2018.pkl")
