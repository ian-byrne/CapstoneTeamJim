# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 12:03:10 2022

@author: melan
"""

import psycopg2
import secrets_melanie
import time
import myutils
import requests
import json
import pandas as pd

# def summary(cur) :
#     total = myutils.queryValue(cur, 'SELECT COUNT(*) FROM swapi;')
#     todo = myutils.queryValue(cur, 'SELECT COUNT(*) FROM swapi WHERE status IS NULL;')
#     good = myutils.queryValue(cur, 'SELECT COUNT(*) FROM swapi WHERE status = 200;')
#     error = myutils.queryValue(cur, 'SELECT COUNT(*) FROM swapi WHERE status != 200;')
#     print(f'Total={total} todo={todo} good={good} error={error}')

# Load the secrets
secrets = secrets_melanie.secrets()

conn = psycopg2.connect(host=secrets['db_url'],
        port=secrets['port'],
        dbname=secrets['db_name'],
        user=secrets['username'],
        password=secrets['password'],
        connect_timeout=10)

cur = conn.cursor()

#%%

print('Create table tax')

print(' ')

sql = '''
CREATE TABLE IF NOT EXISTS state_income_tax (
    State TEXT,
    State_FIPS TEXT,
    High_2000 FLOAT,High_2001 FLOAT,High_2002 FLOAT,High_2003 FLOAT,High_2004 FLOAT,High_2005 FLOAT,High_2006 FLOAT,High_2007 FLOAT,High_2008 FLOAT,High_2009 FLOAT,High_2010 FLOAT,High_2011 FLOAT,High_2012 FLOAT,High_2013 FLOAT,
    High_2014 FLOAT,High_2015 FLOAT,High_2016 FLOAT,High_2017 FLOAT,High_2018 FLOAT,High_2019 FLOAT,High_2020 FLOAT,High_2021 FLOAT,High_2022 FLOAT,
    Low_2000 FLOAT,Low_2001 FLOAT,Low_2002 FLOAT,Low_2003 FLOAT,Low_2004 FLOAT,Low_2005 FLOAT,Low_2006 FLOAT,Low_2007 FLOAT,Low_2008 FLOAT,Low_2009 FLOAT,Low_2010 FLOAT,Low_2011 FLOAT,Low_2012 FLOAT,Low_2013 FLOAT,Low_2014 FLOAT,
    Low_2015 FLOAT,Low_2016 FLOAT,Low_2017 FLOAT,Low_2018 FLOAT,Low_2019 FLOAT,Low_2020 FLOAT,Low_2021 FLOAT,Low_2022 FLOAT);'''

print(sql)
cur.execute(sql)
conn.commit()
#%%

copy_sql = '''COPY state_income_tax FROM STDIN WITH CSV HEADER DELIMITER as ',' '''

f = open(r'C:\users\melan\state_income_tax_clean.csv', 'r', encoding = 'utf-8')
# cur.copy_from(f, 'state_income_tax', sep=',')
cur.copy_expert(sql=copy_sql, file=f)
f.close()
conn.commit()
#%%

print('Create table corp_tax')

print(' ')

sql = '''
CREATE TABLE IF NOT EXISTS state_corp_income_tax (
    State TEXT,
    State_FIPS TEXT,
    High_2002 FLOAT,High_2003 FLOAT,High_2004 FLOAT,High_2005 FLOAT,High_2006 FLOAT,High_2007 FLOAT,High_2008 FLOAT,High_2010 FLOAT,High_2011 FLOAT,High_2012 FLOAT,High_2013 FLOAT,
    High_2014 FLOAT,High_2015 FLOAT,High_2016 FLOAT,High_2017 FLOAT,High_2018 FLOAT,High_2019 FLOAT,High_2020 FLOAT,High_2021 FLOAT,High_2022 FLOAT,
    Low_2002 FLOAT,Low_2003 FLOAT,Low_2004 FLOAT,Low_2005 FLOAT,Low_2006 FLOAT,Low_2007 FLOAT,Low_2008 FLOAT,Low_2010 FLOAT,Low_2011 FLOAT,Low_2012 FLOAT,Low_2013 FLOAT,Low_2014 FLOAT,
    Low_2015 FLOAT,Low_2016 FLOAT,Low_2017 FLOAT,Low_2018 FLOAT,Low_2019 FLOAT,Low_2020 FLOAT,Low_2021 FLOAT,Low_2022 FLOAT);'''

print(sql)
cur.execute(sql)
conn.commit()
#%%

copy_sql = '''COPY state_corp_income_tax FROM STDIN WITH CSV HEADER DELIMITER as ',' '''

f = open(r'C:\users\melan\state_corp_income_tax_clean.csv', 'r', encoding = 'utf-8')
# cur.copy_from(f, 'state_income_tax', sep=',')
cur.copy_expert(sql=copy_sql, file=f)
f.close()
conn.commit()

#%%

print('Create table debt ratio')

print(' ')

sql = '''
CREATE TABLE IF NOT EXISTS county_debt_ratio (
    State_FIPS TEXT,
    County_FIPS TEXT,
    High_1999 FLOAT,High_2000 FLOAT,High_2001 FLOAT,High_2002 FLOAT,High_2003 FLOAT,High_2004 FLOAT,High_2005 FLOAT,High_2006 FLOAT,High_2007 FLOAT,High_2008 FLOAT,High_2009 FLOAT,High_2010 FLOAT,High_2011 FLOAT,High_2012 FLOAT,High_2013 FLOAT,
    High_2014 FLOAT,High_2015 FLOAT,High_2016 FLOAT,High_2017 FLOAT,High_2018 FLOAT,High_2019 FLOAT,High_2020 FLOAT,High_2021 FLOAT,
    Low_1999 FLOAT,Low_2000 FLOAT,Low_2001 FLOAT,Low_2002 FLOAT,Low_2003 FLOAT,Low_2004 FLOAT,Low_2005 FLOAT,Low_2006 FLOAT,Low_2007 FLOAT,Low_2008 FLOAT,Low_2009 FLOAT,Low_2010 FLOAT,Low_2011 FLOAT,Low_2012 FLOAT,Low_2013 FLOAT,Low_2014 FLOAT,
    Low_2015 FLOAT,Low_2016 FLOAT,Low_2017 FLOAT,Low_2018 FLOAT,Low_2019 FLOAT,Low_2020 FLOAT,Low_2021 FLOAT);'''

print(sql)
cur.execute(sql)
conn.commit()
#%%

copy_sql = '''COPY county_debt_ratio FROM STDIN WITH CSV HEADER DELIMITER as ',' '''

f = open(r'C:\users\melan\debt_ratio.csv', 'r', encoding = 'utf-8')
# cur.copy_from(f, 'state_income_tax', sep=',')
cur.copy_expert(sql=copy_sql, file=f)
f.close()
conn.commit()

#%%
conn.commit()
cur.close()