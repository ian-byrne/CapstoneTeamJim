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

    # load data frames
    preds = pd.read_csv("streamlit/data/pytorch_monthly2021_preds.csv")
    preds = preds[preds["property_type"] == "All Residential"]
    preds["diff"] = preds["diff"] / preds["target"]

    st.header("Temporal Fusion Transformer")
    st.write(
        """
    Another option we looked at was using a neural network. Transformers such as BERT
    and GPT have been the cutting edge options for NLP recently. There has been
    considerable research into applying this same idea to time series since much like
    text, time series follow a specific sequence. While researching potential options,
    we came across the Temporal Fusion Transformer model, which is specifically designed
     for multi horizon predictions. This seemed to be a great model candidate due to the
     recent results that it has produced against other cutting edge time series
     prediction models. With the datasets used by the authors, they improved results by
     between 3% and 26% over the next best alternative.[cite]"""
    )

    st.write(
        """
    The benefit of using a transformer style architecture with sequential data is its
    ability to reference much earlier period or words in a sequence without the overhead
    that a typical RNN or LSTM network would have. While both of those models can look
    back at older values, as the data sequence becomes longer those references can get
    “watered down” and add significant computing cost. The transformer avoids this by
    using what is called attention, an architecture introduced in 2017 with the seminal
    paper “Attention is all you need” from Google Research[cite]. Attention can be
    boiled down to “what part of the input sequence should the model focus on.” Using a
    query, key, value system, the model is able to look back at specific variables and
    their positional encoding. This is not unlike a query, key, and value in the sense
    of a database, where one would have a question, the question would then match a key,
    and that value would be what the model paid attention to for that iteration."""
    )
    ##############################################################
    # TODO: Data sections

    ##############################################################

    ##############################################################

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
    st.write("Below is a table of how the model scored using the yearly data")
    yearly_scores = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Metric", "Model", "Baseline"],
                    line_color="darkslategray",
                    fill_color="#3366CC",
                    align=["center", "center"],
                    font=dict(color="white", size=20),
                    height=40,
                ),
                cells=dict(
                    values=[
                        res_v_base_yr["Metric"],
                        res_v_base_yr["Model"],
                        res_v_base_yr["Baseline"],
                    ],
                    line_color="darkslategray",
                    fill=dict(color=["white", "white"]),
                    align=["center", "center"],
                    font_size=18,
                    height=30,
                ),
            )
        ]
    )

    yearly_scores.update_layout(height=400)
    st.plotly_chart(yearly_scores, use_container_width=True)

    st.write("Below is a table of how the model scored using the monthly data")

    monthly_scores = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Metric", "Model", "Baseline"],
                    line_color="darkslategray",
                    fill_color="#3366CC",
                    align=["center", "center"],
                    font=dict(color="white", size=20),
                    height=40,
                ),
                cells=dict(
                    values=[
                        res_v_base_mo["Metric"],
                        res_v_base_mo["Model"],
                        res_v_base_mo["Baseline"],
                    ],
                    line_color="darkslategray",
                    fill=dict(color=["white", "white"]),
                    align=["center", "center"],
                    font_size=18,
                    height=30,
                ),
            )
        ]
    )

    monthly_scores.update_layout(height=400)
    st.plotly_chart(monthly_scores, use_container_width=True)

    ###############################################################
    pred_error = px.choropleth(
        preds,
        geojson=counties,
        locations="county_fips",
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
    # PITA to get predictions from this model -  docs not clear at all
    ###############################################################
    # TODO: Conclusion of the model
    st.write("Conclusion...")

    ###############################################################
    # TODO: References

    with st.expander("References"):
        st.write(
            """
        - References go here.
        """
        )
