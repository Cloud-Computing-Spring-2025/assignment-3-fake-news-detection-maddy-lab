from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import lower, col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
import pandas as pd

# Step 1: Start Spark session
spark = SparkSession.builder \
    .appName("FakeNewsClassification_Task5") \
    .getOrCreate()

# Step 2: Load dataset & preprocess again (for evaluation reuse)
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

df_lower = df.withColumn("text", lower(col("text")))
tokenizer = Tokenizer(inputCol="text", outputCol="words_token")
df_tokenized = tokenizer.transform(df_lower)
remover = StopWordsRemover(inputCol="words_token", outputCol="filtered_words")
df_filtered = remover.transform(df_tokenized)

hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
featurizedData = hashingTF.transform(df_filtered)

idf = IDF(inputCol="raw_features", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

indexer = StringIndexer(inputCol="label", outputCol="label_index")
indexedData = indexer.fit(rescaledData).transform(rescaledData)

# Step 3: Train/test split
train_data, test_data = indexedData.randomSplit([0.8, 0.2], seed=42)

# Step 4: Train logistic regression
lr = LogisticRegression(featuresCol="features", labelCol="label_index")
lr_model = lr.fit(train_data)

# Step 5: Predictions
predictions = lr_model.transform(test_data)

# Step 6: Evaluate
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_acc.evaluate(predictions)

evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="f1")
f1_score = evaluator_f1.evaluate(predictions)

# Step 7: Save to CSV (using pandas for markdown style)
results_pd = pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score"],
    "Value": [round(accuracy, 4), round(f1_score, 4)]
})
results_pd.to_csv("task5_output.csv", index=False)

print("✅ task5_output.csv has been saved with evaluation metrics.")
