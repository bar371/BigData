import numpy as np
import pandas as pd
import os
# from pyspark.ml.feature import Imputer
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import databricks.koalas as ks
import spark_sklearn

from data.evaluation import get_model_stats
from utils import *
from pyspark.ml import Pipeline

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
import operator
# sc.setLogLevel(0)

ks.set_option('compute.ops_on_diff_frames', True)
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("spark bigdata") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")


from sklearn.model_selection import train_test_split

def create_train_val_sets(df):
    print('splitting data...')
    splits = df.randomSplit([0.8, 0,2])
    return splits[0], splits[1]
    # return train_test_split(df, test_size=0.2)

def baseline(train:ks.DataFrame, val:ks.DataFrame):
    max_prob_class = max(train['high_fare'].value_counts().items(), key=operator.itemgetter(1))[0]
    simple_acc = len(val[val['high_fare'] == max_prob_class]) / len(val)
    print("The simplest prediciton will be the max prior on test, {} , which gives us {} acc".format(max_prob_class, simple_acc))

required_features = ['Pclass',
                     'Age',
                     'Fare',
                     'Gender',
                     'Boarded'
                     ]

def train_and_eval(model, train, val):
    trained_model = model.fit(train)
    get_model_stats(trained_model, val)

def model_factory(model_choice, features, train):
    from pyspark.ml.feature import VectorAssembler
    assembler = VectorAssembler(inputCols=features, outputCol='features')
    transformed_data = assembler.transform(train)
    model = None
    if model_choice == LR:
        model = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    assert(model)
    return model

if __name__ == '__main__':
    df = read_pickle(TRAIN_PATH_PICKLE_SMALL)
    print(len(df))
    df = ks.from_pandas(df).to_spark() # we will not need to this conversation later
    print('done conversion')
    train_df, val_df = create_train_val_sets(df)

    # # baseline(train_df, val_df)
    print('loading model..')
    model = model_factory(LR)
    print(model)
    print("begin training")
    train_and_eval(model, train_df, val_df)
