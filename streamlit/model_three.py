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
    # TODO: filter the dataframe to "all residential"
    preds = pd.read_csv("streamlit/data/pytorch_monthly2021_preds.csv")

    st.header("Temporal Fusion Transformer ")
    # TODO: Intro write up
    st.write("Intro Text")

    # TODO: Overview of tft
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
    res_v_base_yr = pd.DataFrame(
        {
            "Metric": ["MAE", "RMSE", "R^2"],
            "Model": [41443.996, 98713.62, 0.729],
            "Baseline": [59424.094, 113088.3, 0.645],
        }
    )
    res_v_base_mo = pd.DataFrame(
        {
            "Metric": ["MAE", "RMSE", "R^2"],
            "Model": [42026.887, 115940.914, 0.787],
            "Baseline": [54858.941, 159162.2, 0.599],
        }
    )

    ###############################################################
    # TODO: 2021 test results chloropleth

    pred_error = px.choropleth(
        preds,
        geojson=counties,
        locations="FIPS",
        color="diff",
        color_continuous_scale="Viridis",
        hover_name="region",
        hover_data=["preds"],
        scope="usa",
        labels={"diff": "2021 Difference between Prediction & Target"},
        title="Average forecast error by ACS county for 2021 prediction",
    )
    pred_error.update_layout(height=500)
    st.plotly_chart(pred_error, use_container_width=True)

    ###############################################################
    # TODO: 2022 predictions if time permits

    ###############################################################
    # TODO: Conclusion of the model

    ###############################################################
    # TODO: References

    with st.expander("References"):
        st.write(
            """
        - References go here.
        """
        )
