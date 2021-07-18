import numpy as np
import pandas as pd
import os
# from pyspark.ml.feature import Imputer
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import databricks.koalas as ks
from utils import *

ks.set_option('compute.ops_on_diff_frames', True)
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("spark bigdata") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


def load_data(data_path:str):
    ## TODO switch to ks...
    print('loading data...')
    df = ks.read_csv(data_path)
    print('done.')
    return df


def haversine_distance(row):
    lat_p, lon_p = row['pickup_latitude'], row['pickup_longitude']
    lat_d, lon_d = row['dropoff_latitude'], row['dropoff_longitude']
    radius = 6371 # km
    dlat = np.radians(lat_d - lat_p)
    dlon = np.radians(lon_d - lon_p)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat_p)) * np.cos(np.radians(lat_d)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = radius * c
    return distance

def pre_process(df):
    print('starting preprocess....')
    print(' High Fare setting..')
    df['high_fare'] = df['fare_amount'].mask(df['fare_amount'] <= 10, other=0)
    df['high_fare'] = df['high_fare'].mask(df['high_fare'] > 10, other=1)
    print(' Removing Nans')
    df.dropna(inplace=True) # removes all nas
    df = df[df['fare_amount'] < 500].dropna() # in eda we found above 500 to be outlier s
    df = df[df['fare_amount'] > 0].dropna() # in eda we found negative fares (should we remove those?)
    print(" Calculating distance (takes a while)")
    df['distance'] = df.apply(haversine_distance, axis=1) # calc distance
    print(" create some seasonal features")
    df['pickup_datetime'] = ks.to_datetime(df['pickup_datetime'])
    df['day_in_month'] = df['pickup_datetime'].dt.day
    df['day_in_week'] = df['pickup_datetime'].dt.day_name()
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year
    df['hour'] = df['pickup_datetime'].dt.hour
    df['is_weekend'] = df['day_in_week'].apply( lambda x: True if x in ['Friday', 'Saturday', 'Sunday'] else False)
    holidays = calendar().holidays(start=df['pickup_datetime'].min(), end=df['pickup_datetime'].max())
    df['holiday'] = df['pickup_datetime'].isin(holidays)
    print(' Preprocess Done!')
    return df

def create_small_trainset():
    df = read_pickle('df_train_post.pkl')
    df = df.sample(frac=0.0001)
    dump_to_pickle(df, 'df_train_small.pkl')

if __name__ == '__main__':
    # create_small_trainset()
    df_train = load_data(TRAIN_PATH)
    # print(df_train.columns)
    df_train = df_train.sample(frac=0.0001)
    pre_process(df_train)
    dump_to_pickle(df_train, 'train_df_ks_tiny.pkl')
    print("Finished")