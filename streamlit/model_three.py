"""Script designed to display the pytorch model write up for Team JIM Capstone."""
# standard imports
import streamlit as st
import pandas as pd
import numpy as np
from urllib.request import urlopen
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

    preds = pd.read_csv("streamlit/data/pytorch_monthly2021_preds.csv")

    st.header("Temporal Fusion Transformer ")

    st.write("Intro Text")
    ##############################################################

    st.write("Below is the histogram of residual error")

    hist = px.histogram(preds["diff"], title="Histogram of residual error of model")

    st.plotly_chart(hist, use_container_width=True)

    ###############################################################
    st.write("Below are the prediction errors for all counties")

    pred_errors = px.scatter(
        preds,
        x="target",
        y="diff",
        title="Residual Error Plot",
        labels={"target": "True Median Sale Price", "diff": "Residual Error"},
        template="plotly_white",
    )
    pred_errors.update_traces(marker_color="#3366CC")
    st.plotly_chart(pred_errors, use_container_width=True)

    ###############################################################
    # TODO: add results from yearly and monthly models here, make df
    # res_v_base = pd.DataFrame()

    ###############################################################
    # TODO: 2021 test results chloropleth

    ###############################################################
    # TODO: 2020 predictions if time permits

    ###############################################################

    ###############################################################
    # TODO: References

    with st.expander("References"):
        st.write(
            """
        - References go here.
        """
        )
