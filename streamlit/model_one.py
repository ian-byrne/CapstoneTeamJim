"""Example Model file."""
import streamlit as st
from utils import load_data


def model1():
        
    st.write("Model 1 running...Testing Streamlit interface - here is the Supervised analysis")

    
    
    st.header("Extra Trees Regressor  :deciduous_tree: 	")
    
    st.write("Below is the predictions for all the available counties")

    # Insert plotly plots

    # st.plotly_chart(Pred2022_plot,use_container_width=True)


    st.header("Top 10 Counties from Extra Trees Regressor model  :evergreen_tree:")
    st.write("Below is the top 10 predicted counties for model comparison")


    # Insert plotly plots

    # st.plotly_chart(Pred2022_plot,use_container_width=True)