# from pyspark.ml.feature import Imputer
import databricks.koalas as ks

from evaluation import get_model_stats
from utils import *

from pyspark.ml.classification import LogisticRegression
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

#
# def create_train_val_sets(df):
#     print('splitting data...')
#     splits = df.randomSplit([0.8, 0,2])
#     return splits[0], splits[1]
    # return train_test_split(df, test_size=0.2)

def baseline(train:ks.DataFrame, val:ks.DataFrame):
    max_prob_class = max(train['high_fare'].value_counts().items(), key=operator.itemgetter(1))[0]
    simple_acc = len(val[val['high_fare'] == max_prob_class]) / len(val)
    print("The simplest prediciton will be the max prior on test, {} , which gives us {} acc".format(max_prob_class, simple_acc))


def model_factory_train(model_choice, features, df):
    from pyspark.ml.feature import VectorAssembler
    assembler = VectorAssembler(inputCols=features, outputCol='features', labelCol='high_fare')
    transformed_data = assembler.transform(df)
    train , val = transformed_data.randomSplit([0.8 , 0.2])
    model = None
    if model_choice == LR:
        model = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    assert(model)
    model.fit(train)
    get_model_stats(model, train)
    get_model_stats(model, val)
    return model

if __name__ == '__main__':
    df = read_pickle(TRAIN_PATH_PICKLE_SMALL)
    # print(len(df))
    print(df.columns)
    df = ks.from_pandas(df).to_spark() # we will not need to this conversation later
    print('done conversion')
    # train_df, val_df = create_train_val_sets(df)

    # # baseline(train_df, val_df)
    print('loading model..')
    features = ['fare_amount', 'pickup_datetime', 'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count']
    model = model_factory_train(LR, features, df)
    print(model)
    print("begin training")
