from pyspark.sql import SparkSession
import os
import shutil

# Step 1: Create Spark session
spark = SparkSession.builder \
    .appName("FakeNewsClassification_Task1") \
    .getOrCreate()

# Step 2: Load the CSV file
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Step 3: Create a temporary view
df.createOrReplaceTempView("news_data")

# Step 4a: Show the first 5 rows
print("First 5 rows:")
df.show(5)

# Step 4b: Count total number of articles
print(f"Total number of articles: {df.count()}")

# Step 4c: Distinct labels
print("Distinct labels:")
df.select("label").distinct().show()

# Step 5: Save first 5 rows to a CSV file as task1_output.csv
# First write to a temporary output directory
output_dir = "task1_output"
df.limit(5).coalesce(1).write.csv(output_dir, header=True, mode="overwrite")

# Then rename the part file to task1_output.csv and clean up
for filename in os.listdir(output_dir):
    if filename.startswith("part-") and filename.endswith(".csv"):
        shutil.move(os.path.join(output_dir, filename), "task1_output.csv")
        break
shutil.rmtree(output_dir)

print("✅ task1_output.csv has been saved.")
