
# 📰 Fake News Detection using Apache Spark MLlib

This project walks through a **complete machine learning pipeline** using Apache Spark to classify news articles as **FAKE** or **REAL** based on their textual content. The project utilizes PySpark's MLlib components for large-scale text preprocessing, feature extraction, model training, and evaluation.

---

## 📁 Dataset Description

**Filename:** `fake_news_sample.csv`

**Columns:**
- `id`: Unique identifier for each news article.
- `title`: Title of the article.
- `text`: Full content of the article.
- `label`: Target classification (either `FAKE` or `REAL`).

---

## 🚀 Environment Setup

1. Install Spark and PySpark:

```bash
pip install pyspark
```

2. Make sure you're running inside a Spark-compatible environment. Use `spark-submit` or Python to execute each task file.

---

## ✅ TASK 1: Load and Explore the Dataset

### Goals:
- Read CSV file into a Spark DataFrame
- Create a temporary SQL view
- Run basic exploratory queries

### Code Highlights:
```python
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)
df.createOrReplaceTempView("news_data")

# Show first 5 rows
df.show(5)

# Total number of articles
print(df.count())

# Distinct labels
df.select("label").distinct().show()
```

### Output File:
- `task1_output.csv`: Sample of 5 rows from the dataset.

---

## 🧹 TASK 2: Text Preprocessing

### Goals:
- Convert text to lowercase
- Tokenize the text
- Remove stopwords

### Code Highlights:
```python
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import lower

df = df.withColumn("text", lower(df["text"]))

tokenizer = Tokenizer(inputCol="text", outputCol="words_token")
df_tokenized = tokenizer.transform(df)

remover = StopWordsRemover(inputCol="words_token", outputCol="filtered_words")
df_cleaned = remover.transform(df_tokenized)
```

### Note:
- `filtered_words` is an array column, so we convert it to a string before saving to CSV.

```python
from pyspark.sql.functions import col, concat_ws
df_result = df_cleaned.select("id", "title", concat_ws(" ", "filtered_words").alias("filtered_words"), "label")
```

### Output File:
- `task2_output.csv`: Cleaned and tokenized text.

---

## ✨ TASK 3: Feature Extraction

### Goals:
- Convert words to numerical vectors using TF-IDF
- Index labels using StringIndexer
- Assemble features

### Code Highlights:
```python
from pyspark.ml.feature import HashingTF, IDF, StringIndexer

hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features")
df_featurized = hashingTF.transform(df_cleaned)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df_featurized)
df_tfidf = idf_model.transform(df_featurized)

indexer = StringIndexer(inputCol="label", outputCol="label_index")
df_final = indexer.fit(df_tfidf).transform(df_tfidf)
```

### Output File:
- `task3_output.csv`: Includes TF-IDF features and indexed labels.

---

## 🤖 TASK 4: Model Training and Prediction

### Goals:
- Split data into training and testing sets
- Train a Logistic Regression model
- Predict labels on the test set

### Code Highlights:
```python
from pyspark.ml.classification import LogisticRegression

train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=42)

lr = LogisticRegression(featuresCol="features", labelCol="label_index")
model = lr.fit(train_data)

predictions = model.transform(test_data)
```

### Output File:
- `task4_output.csv`: Includes original label and predicted label.

---

## 📊 TASK 5: Model Evaluation

### Goals:
- Evaluate the performance of the model using:
  - Accuracy
  - F1 Score

### Code Highlights:
```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator_acc = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="f1")

accuracy = evaluator_acc.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)
```

### Output File:
- `task5_output.csv`

### Sample Metrics:
```markdown
| Metric   |   Value |
|:---------|--------:|
| Accuracy |       1 |
| F1 Score |       1 |
```

---

## 📂 Project Structure

```
├── fake_news_sample.csv
├── task1_task.py
├── task2_task.py
├── task3_task.py
├── task4_task.py
├── task5_task.py
├── task1_output.csv
├── task2_output.csv
├── task3_output.csv
├── task4_output.csv
├── task5_output.csv
└── README.md
```

---

## 🧠 Key Learnings

- Working with real-world unstructured text data
- Cleaning and transforming data for ML
- Building an end-to-end classification pipeline using PySpark
- Evaluating model performance on unseen data

---

## 👨‍💻 Author

Built with ❤️ using Apache Spark by [Your Name]  
Spring 2025 — UNC Charlotte

