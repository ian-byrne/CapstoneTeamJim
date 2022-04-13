import psycopg2
import configparser
import pandas as pd

def connect(name):
    config_obj = configparser.ConfigParser()
    config_obj.read("config.ini")
    name = config_obj[name]

    user = name["user"]
    password = name["password"]
    host = name["host"]
    dbname = name["dbname"]
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
    return conn


