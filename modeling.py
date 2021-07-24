# from pyspark.ml.feature import Imputer

### imports...
import databricks.koalas as ks
from pyspark.ml.classification import LogisticRegression
import operator
# sc.setLogLevel(0)
from evaluation import get_model_stats
from utils import LR, read_pickle, TRAIN_PATH_PICKLE_SMALL, TRAIN_PATH_PICKLE_TINY_PD
ks.set_option('compute.ops_on_diff_frames', True)
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("spark bigdata") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")



def baseline(train:ks.DataFrame, val:ks.DataFrame):
    max_prob_class = max(train['high_fare'].value_counts().items(), key=operator.itemgetter(1))[0]
    simple_acc = len(val[val['high_fare'] == max_prob_class]) / len(val)
    print("The simplest prediciton will be the max prior on test, {} , which gives us {} acc".format(max_prob_class, simple_acc))


def model_factory_train(model_choice, features, df):
    from pyspark.ml.feature import VectorAssembler
    assembler = VectorAssembler(inputCols=features, outputCol='features')
    transformed_data = assembler.transform(df)
    train , val = transformed_data.randomSplit([0.8 , 0.2])
    model = None
    if model_choice == LR:
        model = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, featuresCol='features',labelCol='high_fare')
    assert(model)
    ret = model.fit(train)
    predict_train = ret.transform(train)
    predict_val = ret.transform(val)

    # print(predict_val.select("high_fare", "prediction")) # just testing we can predict

    get_model_stats(predict_train)
    # get_model_stats(model, val)
    return model

if __name__ == '__main__':
    df = read_pickle(TRAIN_PATH_PICKLE_TINY_PD)
    # print(len(df))
    print(df.columns)
    ## these columns we do not wish to keep for modeling
    to_drop = ['key', 'fare_amount', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'pickup_datetime','day_in_week']
    df.drop(to_drop,axis=1, inplace=True)

    print(df.columns)
    df = spark.createDataFrame(df)
    # df = ks.from_pandas(df).to_spark() # we will not need to this conversation later
    print('done conversion')
    # train_df, val_df = create_train_val_sets(df)

    # # baseline(train_df, val_df)

    features_to_train = ['passenger_count',
     'day_in_month', 'is_weekend', 'day_of_week_is_Friday',
     'day_of_week_is_Monday', 'day_of_week_is_Saturday',
     'day_of_week_is_Sunday', 'day_of_week_is_Thursday',
     'day_of_week_is_Tuesday', 'day_of_week_is_Wednesday', 'month', 'year',
     'hour', 'holiday', 'distance']

    print("begin training")
    model = model_factory_train(LR, features_to_train, df)
