from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.sql.functions import lower, col, concat_ws
import os
import shutil

# Step 1: Start Spark session
spark = SparkSession.builder \
    .appName("FakeNewsClassification_Task3") \
    .getOrCreate()

# Step 2: Load dataset
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Step 3: Preprocess - lowercase, tokenize, remove stopwords
df_lower = df.withColumn("text", lower(col("text")))
tokenizer = Tokenizer(inputCol="text", outputCol="words_token")
df_tokenized = tokenizer.transform(df_lower)
remover = StopWordsRemover(inputCol="words_token", outputCol="filtered_words")
df_filtered = remover.transform(df_tokenized)

# Step 4: TF-IDF features
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
featurizedData = hashingTF.transform(df_filtered)

idf = IDF(inputCol="raw_features", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Step 5: Label indexing
indexer = StringIndexer(inputCol="label", outputCol="label_index")
indexedData = indexer.fit(rescaledData).transform(rescaledData)

# Step 6: Convert incompatible columns to string (for CSV)
df_result = indexedData.select(
    "id",
    concat_ws(" ", col("filtered_words")).alias("filtered_words_str"),
    col("features").cast("string").alias("features_str"),
    "label_index"
)

# Step 7: Save to temporary output directory
output_dir = "task3_output"
df_result.coalesce(1).write.csv(output_dir, header=True, mode="overwrite")

# Step 8: Rename the part file and clean up
for filename in os.listdir(output_dir):
    if filename.startswith("part-") and filename.endswith(".csv"):
        shutil.move(os.path.join(output_dir, filename), "task3_output.csv")
        break
shutil.rmtree(output_dir)

print("✅ task3_output.csv has been saved successfully.")
