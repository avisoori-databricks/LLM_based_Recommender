# Databricks notebook source
# MAGIC %md
# MAGIC ### Evaluation
# MAGIC For the purpose of evaluation, we need a dataset with past purchase history which we do not have with the product description data. We instead use the Instacart dataset from the Market Basket challenge on Kaggle. However the process for evaluating is given here for the sake of completion because this data lacks product descriptions to optimally fit the proposed Generation Augmented Retrieval (GAR as opposed to RAG). 

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch-preview
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
client = VectorSearchClient()

# COMMAND ----------

# MAGIC %md 
# MAGIC You can download the instacart data from here: https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset. Unzip and upload to the purchase_raw Unity Catalog volume that was created earlier. Instructions for uploading the Volumes can be found here: https://docs.databricks.com/en/ingestion/add-data/upload-to-volume.html#:~:text=In%20the%20sidebar%2C%20click%20New,File%20%3E%20Upload%20files%20to%20Volume.
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- CREATE A CATALOG, SCHEMA AND VOLUME TO STORE DATA NEEDED FOR THIS. IN PRACTICE, YOU COULD USE AN EXISTING VOLUME
# MAGIC CREATE CATALOG IF NOT EXISTS llm_recommender;
# MAGIC USE CATALOG llm_recommender;
# MAGIC CREATE DATABASE IF NOT EXISTS purchase_data;
# MAGIC USE DATABASE purchase_data;
# MAGIC CREATE VOLUME IF NOT EXISTS purchase_raw;

# COMMAND ----------

df = spark.read.option("header", True).csv('/Volumes/llm_recommender/purchase_data/purchase_raw/products.csv').selectExpr("product_id as id","product_name as product")
display(df)

# COMMAND ----------

df.write.saveAsTable('product_info')

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE product_info SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

client.create_index(
  source_table_name="llm_recommender.purchase_data.product_info",
  dest_index_name="vs_catalog.vs_schema.instacart-grocery-product-index",
  primary_key="id",
  index_column="product",
  embedding_model_endpoint_name="all-MiniLM-L6-v2-avi")

# COMMAND ----------

index = client.list_indexes("vs_catalog")
index

# COMMAND ----------

