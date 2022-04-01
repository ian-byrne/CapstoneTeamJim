"""Demo app for streamlit."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import psycopg2
from secrets_ian import capstone_db

# db conn
engine = capstone_db["engine"]
db_name = capstone_db["db_name"]
user = capstone_db["username"]
pw = capstone_db["password"]
host = capstone_db["db_url"]

conn = psycopg2.connect(dbname=db_name, user=user, password=pw, host=host)

# basic writing to the page
st.set_page_config(page_title="CapstoneProject")
st.write("## Hello, Capstone team JIM!!")
st.write("Potential for the team to utilize streamlit as a interactive essay.")

# input examples for potential interactivity
variable = st.text_input("Enter variable")
st.write(variable)

number = st.number_input("Enter number")  # can set max and min if needed as well
st.write(number)

# display dataframe
cols = [
    "county_fips",
    "period_end",
    "property_type",
    "property_type_id",
    "median_sale_price_yoy",
    "median_list_price_yoy",
    "median_ppsf_yoy",
    "median_list_ppsf_yoy",
    "homes_sold_yoy",
    "pending_sales_yoy",
    "new_listings_yoy",
    "inventory_yoy",
    "months_of_supply_yoy",
    "median_dom_yoy",
    "avg_sale_to_list_yoy",
    "sold_above_list_yoy",
    "price_drops_yoy",
    "off_market_in_two_weeks_yoy",
]

rf_sql = """
select
*
from redfin_county_full
limit 500;
"""

redfin = pd.read_sql(rf_sql, con=conn)
to_display = redfin.head()

st.dataframe(to_display)

# display plotly chart


def main():
    """Run Main."""
    pass


if __name__ == "__main__":
    main()

