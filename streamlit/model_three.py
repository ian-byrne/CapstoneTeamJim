"""Script designed to display the pytorch model write up for Team JIM Capstone."""
# standard imports
from re import I
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio


def pytorch_writeup():
    """Display the pytorch-forecasting write up."""
    with urlopen(
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    ) as response:
        counties = json.load(response)

    preds = pd.read_csv("streamlit/data/monthly2021_pytorch_pred_allresidential.csv")

    st.header("Temporal Fusion Transformer ")
    st.write("Below is the histogram of residual error")

    hist = px.histogram(
        preds["target-prediction"], title="Histogram of residual error of model"
    )

    st.plotly_chart(hist, use_container_width=True)

    st.write("Below is the predictions error for all counties")

