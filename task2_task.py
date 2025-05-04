from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import lower, col, concat_ws
import os
import shutil

# Step 1: Create Spark session
spark = SparkSession.builder \
    .appName("FakeNewsClassification_Task2") \
    .getOrCreate()

# Step 2: Load the CSV
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Step 3: Convert text to lowercase
df_lower = df.withColumn("text", lower(col("text")))

# Step 4: Tokenize text
tokenizer = Tokenizer(inputCol="text", outputCol="words_token")
df_tokenized = tokenizer.transform(df_lower)

# Step 5: Remove stopwords
remover = StopWordsRemover(inputCol="words_token", outputCol="filtered_words")
df_filtered = remover.transform(df_tokenized)

# Step 6: Convert array of words to space-separated string for CSV compatibility
df_result = df_filtered.select(
    "id",
    "title",
    concat_ws(" ", "filtered_words").alias("filtered_text"),
    "label"
)

# Step 7: Save to CSV
output_dir = "task2_output"
df_result.coalesce(1).write.csv(output_dir, header=True, mode="overwrite")

# Step 8: Rename part file to task2_output.csv and delete the folder
for filename in os.listdir(output_dir):
    if filename.startswith("part-") and filename.endswith(".csv"):
        shutil.move(os.path.join(output_dir, filename), "task2_output.csv")
        break
shutil.rmtree(output_dir)

print("✅ task2_output.csv has been saved successfully.")
