from secrets_ian import capstone_db
import pandas as pd
import psycopg2

engine = capstone_db["engine"]
db_name = capstone_db["db_name"]
user = capstone_db["username"]
pw = capstone_db["password"]
host = capstone_db["db_url"]


def db_conn():
    conn = psycopg2.connect(dbname=db_name, user=user, password=pw, host=host)
    return conn
