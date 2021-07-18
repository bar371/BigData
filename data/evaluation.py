from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics

# Load training data in LIBSVM format

# Run training algorithm to build the model
def get_model_stats(model, test):
    # Compute raw scores on the test set
    predictionAndLabels = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))

    # Instantiate metrics object
    metrics = MulticlassMetrics(predictionAndLabels)
    # Overall statistics
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1Score = metrics.fMeasure(1.0)
    print("Summary Stats")
    print("Precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)

    # # Statistics by class
    # labels = data.map(lambda lp: lp.label).distinct().collect()
    # for label in sorted(labels):
    #     print("Class %s precision = %s" % (label, metrics.precision(label)))
    #     print("Class %s recall = %s" % (label, metrics.recall(label)))
    #     print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))
    #
    # # Weighted stats
    # print("Weighted recall = %s" % metrics.weightedRecall)
    # print("Weighted precision = %s" % metrics.weightedPrecision)
    # print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
    # print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
    # print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)