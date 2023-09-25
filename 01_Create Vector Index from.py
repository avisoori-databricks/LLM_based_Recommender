# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch-preview
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from datasets import load_dataset , Dataset, concatenate_datasets 
import pandas as pd

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
client = VectorSearchClient()

# COMMAND ----------

# MAGIC %sql
# MAGIC -- CREATE A CATALOG, SCHEMA AND VOLUME TO STORE DATA NEEDED FOR THIS. IN PRACTICE, YOU COULD USE AN EXISTING VOLUME
# MAGIC CREATE CATALOG IF NOT EXISTS llm_recommender;
# MAGIC USE CATALOG llm_recommender;
# MAGIC CREATE DATABASE IF NOT EXISTS purchase_data;
# MAGIC USE DATABASE purchase_data;
# MAGIC CREATE VOLUME IF NOT EXISTS purchase_raw;

# COMMAND ----------

prod_ds = load_dataset("xiyuez/red-dot-design-award-product-description")
prod_df = pd.DataFrame(prod_ds['train'])
display(prod_df)

# COMMAND ----------

prod_df['id'] = prod_df.reset_index().index
df = prod_df[['id', 'text']]
display(df)

# COMMAND ----------

spark.createDataFrame(df).write.saveAsTable('product_descriptions')

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE product_descriptions SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

client.create_index(
  source_table_name="llm_recommender.purchase_data.product_descriptions",
  dest_index_name="vs_catalog.vs_schema.reddot-product-index",
  primary_key="id",
  index_column="text",
  embedding_model_endpoint_name="all-MiniLM-L6-v2-avi")

# COMMAND ----------

index = client.list_indexes("vs_catalog")
index

# COMMAND ----------

client.get_index('vs_catalog.vs_schema.reddot-product-index')

# COMMAND ----------

client.similarity_search(
  index_name = "vs_catalog.vs_schema.reddot-product-index",
  query_text = "I want to buy a toaster",
  columns = ["id", "text"], # columns to return
  num_results = 5)

# COMMAND ----------

