# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 08:09:44 2022

@author: melan
"""

import psycopg2
import secrets_melanie
import pandas as pd
import numpy as np
import pickle



def connect_db(login):
    
    if login!= None:
        
            conn = psycopg2.connect(host="mads-capstone-db.codt5pemnxi8.us-east-1.rds.amazonaws.com",
                    port=5432,
                    dbname="capstone",
                    user=login['username'],
                    password=login['password'],
                    connect_timeout=10)
    else:
        print("You don't have access to the database.  Please contact database admin Ian Byrne")
    
    return conn

def extract(connection):

       
    # cur = connection.cursor()    

    # Redfin 
    query = """
    select 
    county_fips,
    period_end, 
    property_type, 
    median_sale_price,
    median_sale_price_mom,
    median_ppsf,
    median_ppsf_mom
    
    from redfin_county_full
    
    where property_type ='All Residential'
    
    """
    redfin_df = pd.read_sql(query, con=connection)
    connection.close()
    
    return redfin_df
        
def prepare_data(redfin,counties_filter):



    # Create date columns for filtering.  For time series we will use monthly median sale price.
    redfin['date'] = pd.to_datetime(redfin['period_end'])
    redfin['year'] = redfin['date'].dt.year
    redfin['month'] = redfin['date'].dt.month
    
    # Create date df for sequential dates for all county fips
    counties = list(redfin.county_fips.unique())
    years = list(range(2012,2022))
    months =list(range(1,13))
    base = pd.DataFrame([[counties],[years],[months]]).T
    base = base.explode(0)
    base = base.explode(1)
    base = base.explode(2).reset_index(drop=True)
    base.columns=['county_fips','year','month']
    # remove the first month as several counties have this month missing
    base = base[~((base['year']==2012)&(base['month']==1))]
    
    base = base.merge(redfin,how='left',on=['county_fips','year','month'])
    base = base.sort_values(by=['county_fips','year','month']).reset_index(drop=True)

# Using redfin median sale price for time series
    df = base.copy(deep=True)
    df['day']=1
    # time series needs datetime for index
    df['date'] = pd.to_datetime(df[['year','month','day']])
    df['year'] = df['year'].astype(str)
    df = df[['county_fips','date','median_sale_price','year']]
    
    # Determine how many counties have missing values in training df and how many have recent missing
    
    # missing_vals = df[df['median_sale_price'].isna()].fillna(1).groupby(['county_fips'],as_index=False)['median_sale_price'].count()
    # missing_vals.columns=['county_fips','total_missing']
    
    # # cutting off data - starting now at 2016 so checking where missing counties have missing values
    # recent_vals = df[df['date'].dt.year >=2016]
    # recent_vals = recent_vals[recent_vals['median_sale_price'].isna()].fillna(1).groupby(['county_fips'],as_index=False)['median_sale_price'].count()
    # recent_vals.columns=['county_fips','recent_missing']
    # missing_vals = missing_vals.merge(recent_vals, how='left',on='county_fips').fillna(0)
    # missing_vals['missing_pct'] = missing_vals['total_missing']/119*100
    # missing_vals['recent_pct'] = missing_vals['recent_missing']/72*100 #(6 years x 12 months)
    
    # missing_vals['to_filter'] = 1
    # missing_vals['to_filter'][((missing_vals['recent_pct']<10))]=0
    # # missing_counties = missing_vals[missing_vals['to_filter']==1]['county_fips'].to_list()
    # keep_counties = missing_vals[missing_vals['to_filter']==0]['county_fips'].to_list()
    # dropping 467 counties with recent vals <10% in last 6 years.
    with open(counties_filter, 'r') as f:
        lines = f.readlines()
    
    VAR_counties = []
    for line in lines:
        VAR_counties.append(line.strip())
    
    # remove counties with sparse data and fill forward na for median sale price and backfill any values that didn't start in Jan 2016
    df = df[df['county_fips'].isin(VAR_counties)]
    
    return df

def transform_data(df):
    
    df['median_sale_price'] = df.groupby('county_fips')['median_sale_price'].transform(lambda v: v.ffill()).fillna(method="bfill")
    # Predicting on log of median sale price
    df['log_median_sale_price'] = np.log(df['median_sale_price'])
    # split training and testing data and pivot so counties are columns (only using data from 2016 onward)
    df = df[df['year']>='2016']
    df_pivot = df.pivot(index='date',columns='county_fips',values='log_median_sale_price')
    
    # continue to forecast 2021 to extract error for 2022 correction
    df_correction = df[((df['year']>='2016')&(df['year']<'2021'))]
    df_correction_pivot = df_correction.pivot(index='date',columns='county_fips',values='log_median_sale_price')                                                                                          


    return df, df_pivot, df_correction, df_correction_pivot



if __name__ == '__main__':
    import argparse


    secrets = secrets_melanie.secrets()

    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     'new_data', help='Do you want to extract new data from db - respond y or n')
    parser.add_argument(
        'secrets_user', help='Enter username for database login or press enter to bypass and use stored data')
    parser.add_argument(
        'secrets_pwd', help='Enter password for database login or press enter to bypass and use stored data')
    parser.add_argument(
        'filtered_counties', help='Enter filename for counties to filter (TXT')
    parser.add_argument(
        'output_file_main_df', help='the cleaned datafile (PKL)')
    parser.add_argument(
        'output_file_main_df_pivoted', help='the cleaned long datafile (PKL)')
    parser.add_argument(
        'output_file_df_corr', help='the cleaned datafile (PKL)')
    parser.add_argument(
        'output_file_df_corr_pivot', help='the cleaned long datafile (PKL)')
    args = parser.parse_args()

    # use_db = str(args.new_data).lower()
    secrets = {'username':args.secrets_user,
               'password':args.secrets_pwd}
    if secrets['username']==None or secrets['password']==None: 
        #then use historical data file
        print()
    
    counties_to_filter = args.filtered_counties 
    
    #initialize database connection
    db_connection = connect_db(secrets)
    extracted_data = extract(db_connection)
    prepared_data = prepare_data(extracted_data,counties_to_filter)
    
    main_df, main_df_pivoted, df_corr, df_corr_pivot = transform_data(prepared_data)
    main_df.to_pickle(args.output_file_main_df)
    main_df_pivoted.to_pickle(args.output_file_main_df_pivoted)
    df_corr.to_pickle(args.output_file_df_corr)
    df_corr_pivot.to_pickle(args.output_file_df_corr_pivot)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    