from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer
import os
import shutil

# Step 1: Start Spark session
spark = SparkSession.builder \
    .appName("FakeNewsClassification_Task4") \
    .getOrCreate()

# Step 2: Load the dataset again with features column as vector
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Step 3: Preprocessing
from pyspark.sql.functions import lower
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

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

# Step 4: Split the data
train_data, test_data = indexedData.randomSplit([0.8, 0.2], seed=42)

# Step 5: Train the model
lr = LogisticRegression(featuresCol="features", labelCol="label_index")
lr_model = lr.fit(train_data)

# Step 6: Predict on test data
predictions = lr_model.transform(test_data)

# Step 7: Select and save required columns
df_result = predictions.select("id", "title", "label_index", "prediction")

# Step 8: Write to CSV
output_dir = "task4_output"
df_result.coalesce(1).write.csv(output_dir, header=True, mode="overwrite")

# Step 9: Rename part file and clean
for filename in os.listdir(output_dir):
    if filename.startswith("part-") and filename.endswith(".csv"):
        shutil.move(os.path.join(output_dir, filename), "task4_output.csv")
        break
shutil.rmtree(output_dir)

print("✅ task4_output.csv has been saved successfully.")
