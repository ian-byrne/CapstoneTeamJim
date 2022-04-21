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
    f = open("streamlit/data/VAR_counties.txt", "r")
    content = f.read()
    var_counties = content.split("\n")
    f.close()

    var = [int(i) for i in var_counties[1:-1]]
    # load data frames and add proper filters
    preds = pd.read_csv("streamlit/data/pytorch_monthly2021_preds.csv")
    preds = preds[preds["property_type"] == "All Residential"]
    preds["diff"] = preds["diff"] / preds["target"]
    preds = preds[preds["county_fips"].isin(var)]

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
     between 3% and 26% over the next best alternative ([Lim et al., 2019](https://arxiv.org/pdf/1912.09363.pdf))"""
    )

    st.write(
        """
    The benefit of using a transformer style architecture with sequential data is its
    ability to reference much earlier period or words in a sequence without the overhead
    that a typical RNN or LSTM network would have. While both of those models can look
    back at older values, as the data sequence becomes longer those references can get
    “watered down” and add significant computing cost. The transformer avoids this by
    using what is called attention, an architecture introduced in 2017 with the seminal
    paper “Attention is all you need” from Google Research ([Vaswani et al., 2017](https://arxiv.org/pdf/1706.03762.pdf)). Attention can be
    boiled down to “what part of the input sequence should the model focus on.” Using a
    query, key, value system, the model is able to look back at specific variables and
    their positional encoding. This is not unlike a query, key, and value in the sense
    of a database, where one would have a question, the question would then match a key,
    and that value would be what the model paid attention to for that iteration. This
    structure also allows for parallelization which is not available in an RNN or LSTM.
    """
    )
    ##############################################################
    st.subheader("Data selection")
    st.write(
        """
    For the monthly granularity model (the better scoring of the two), the main
    datasets chosen were the Redfin sales data, U.S. monthly unemployment data by
    state, and U.S. state corporate and personal income tax rates. The year over
    year and month over month calculations have been dropped from the monthly model
    Redfin data as the neural network should be able to identify these relationships
    without being explicitly told. All of these data points combined allowed for the
    data to begin in 2012 and run through the end of 2021. The transformer architecture thrives on having more data to work with, hence why I chose to go through 2021. A version of the model was also run on data spanning
    2012-2020 to stay consistent with the other two models in the project, the results of which are shown below. In order to avoid
    potential data leakage, median sale price per square foot was removed as it very
    highly correlated to the target variable of median sale price. Both the yearly
    and the monthly data that are presented here incorporate only the “all
    residential” property type from the Redfin data for consistency with the other two, however, the
    model was trained using all property types.

    A benefit of using the temporal fusion transformer is the ability to classify the
    variables as static or time varying as well as known and unknown. The full data
    points used in the training data are laid out below along with how they were defined
    in the [`TimeSeriesDataSet`](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html) class.

    Below is an example of how the data was defined for the monthly granularity."""
    )

    code = """
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
    """
    st.code(code, language="python")

    st.write(
        """Data points that do not change, such as property type or region are
        classified as static categoricals. Year and time_idx are both categorized as
        time varying real numbers along with the tax data. Classifying tax rates as a
        ime varying known number makes sense due to the fact that when they do change,
        they are generally voted on and known beforehand."""
    )
    ##############################################################
    st.subheader("Hyperparameters")
    st.write(
        """
        Pytorch-forecasting has an integration with a package called [Optuna](https://optuna.org/) that can
        run studies to find the optimal hyperparameters for a specified model. This can
        be thought of as similar to grid search from sklearn. When running this for the
        monthly data, this took just under three hours on my laptop. The study for the
        yearly data took under one hour. The following parameters were searched:
        gradient clip, hidden size, dropout, hidden continuous size, attention head
        size, and learning rate. """
    )
    st.write(
        """
        - For the yearly model the following parameters were suggested: `{
            'gradient_clip_val': 0.012136185295350904,
            'hidden_size': 82,
            'dropout': 0.14907256141821396,
            'hidden_continuous_size': 22,
            'attention_head_size': 1,
            'learning_rate': 0.10036312732505494}`.
        - For the monthly model the following parameters were suggested: `{
            'gradient_clip_val': 0.16539859210424376,
            'hidden_size': 58,
            'dropout': 0.10508905793984676,
            'hidden_continuous_size': 56,
            'attention_head_size': 4,
            'learning_rate': 0.007798356725190109}`.
        """
    )
    st.write(
        """The exact values listed above were not what was used as there was some manual
        tuning done afterwards. It is essential to have manual control over parameters in any neural network.
        For the 2020 predictions presented at the end of this page, monthly data granularity
        was used with the same hyper parameters used in the 2021 model training. """
    )
    ##############################################################
    st.subheader("Errors")
    st.write(
        """Below is the histogram of residual error,  as we can see, the
    majority of the errors are between 0 and 10% below the actual target price. The model
    has developed a bias to predict on the low side of the target."""
    )

    hist = px.histogram(preds["diff"], title="Residual Error of Model")

    st.plotly_chart(hist, use_container_width=True)

    ###############################################################
    st.write("""Below are the prediction errors for all counties.""")

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
    st.subheader("Results with using 2021 as test data.")

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

    st.write(
        """The [Baseline](https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.models.baseline.Baseline.html) model that we compare against in these tables simply
    takes the last observed value of the target and repeats it for however many periods
    it is to forecast out to. Below is a table of how the model scored using the yearly data"""
    )

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
    st.write(
        """As we can see below, the increases at the top end of the model are highly
    unrealistic even in a strong market. It should be noted that most of these projections
    were made in areas that normally have property values under 100k from what we see in
    our data sets."""
    )

    preds2020 = pd.read_csv("streamlit/data/tft_top10_2020.csv")
    preds2020["2019to2020increase"] = preds2020["2019to2020increase"] * 100
    preds2020 = preds2020.round(2)
    top10table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["County", "2020 Price % increase over 2019"],
                    line_color="darkslategray",
                    fill_color="#3366CC",
                    align=["center", "center"],
                    font=dict(color="white", size=20),
                    height=40,
                ),
                cells=dict(
                    values=[preds2020["region"], preds2020["2019to2020increase"]],
                    line_color="darkslategray",
                    fill=dict(color=["white", "white"]),
                    align=["center", "center"],
                    font_size=18,
                    height=30,
                ),
            )
        ]
    )

    top10table.update_layout(width=1000, height=700)
    st.plotly_chart(top10table, use_container_width=True)

    ###############################################################
    st.subheader("Model Conclusions")
    st.write(
        """As we see above, the model can score well depending on the county, but
    the overall scores are lackluster compared to the previous two models used in this
    project. Due to the architecture of transformers and neural networks in
    general, the model would likely perform better if it had more data to train on. By that I
    mean both more years and months to train on, including some recessions and more varied
    pricing periods, as well as more social data such as when lockdowns occurred in each state.
    Large shifts in housing prices were observed once covid lockdowns began which makes
    sense since people were stuck inside all the time and put more thought into their
    home and immediate surroundings. These results can also act as a reminder that sometimes a
    complicated neural network is not the right choice to solve the problem at hand and that
    simpler models can be better."""
    )

    with st.expander("References"):
        st.write(
            """
        1.  [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/pdf/1912.09363.pdf)
        2.  [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
        3.  [Pytorch-Forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/)
        """
        )
