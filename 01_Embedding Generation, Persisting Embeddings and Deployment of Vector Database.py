# Databricks notebook source
#https://github.com/hwchase17/chroma-langchain/blob/master/persistent-qa.ipynb

# COMMAND ----------

# MAGIC %pip install -U chromadb

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id
import chromadb
from chromadb.config import Settings
import time
from chromadb.utils import embedding_functions
from transformers import pipeline
import pandas as pd
from sys import version_info
import cloudpickle
from datasets import load_dataset , Dataset, concatenate_datasets 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- CREATE A CATALOG, SCHEMA AND VOLUME TO STORE DATA NEEDED FOR THIS. IN PRACTICE, YOU COULD USE AN EXISTING VOLUME
# MAGIC CREATE CATALOG IF NOT EXISTS llm_recommender;
# MAGIC USE CATALOG llm_recommender;
# MAGIC CREATE DATABASE IF NOT EXISTS movie_data;
# MAGIC USE DATABASE movie_data;
# MAGIC CREATE VOLUME IF NOT EXISTS movie_lens;
# MAGIC USE CATALOG llm_recommender;

# COMMAND ----------

#persist_directory = "/Volumes/llm_recommender/movie_data/movie_lens/"
persist_directory_dbfs = "/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/product_recsys_chroma"
persist_directory_ephemeral = "/tmp/product_recsys_chroma"

# COMMAND ----------

import os

if not os.path.exists(persist_directory_dbfs):
    os.mkdir(persist_directory_dbfs)

if not os.path.exists(persist_directory_ephemeral):
    os.mkdir(persist_directory_ephemeral)

# COMMAND ----------

prod_ds = load_dataset("xiyuez/red-dot-design-award-product-description")
prod_df = pd.DataFrame(prod_ds['train'])
display(prod_df)
     

# COMMAND ----------

# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.PersistentClient(path=persist_directory_ephemeral)

# COMMAND ----------

prod_df['id'] = prod_df.reset_index().index
ids = prod_df['id'].tolist()
ids =[str(element) for element in ids]
ids[:2]

# COMMAND ----------

docs = prod_df.text.to_list()
#To see if this was done properly 
docs[:3]

# COMMAND ----------

metadata = [{'product': product, 'category': category} for product, category in zip(prod_df['product'].tolist(), prod_df['category'].tolist())]
metadata[:2]

# COMMAND ----------

collection = client.create_collection("product_reviews")

# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.add(
    documents=docs, # we embed for you, or bring your own
    metadatas=metadata, # filter on arbitrary metadata!
    ids=ids, # must be unique for each doc 
)

# COMMAND ----------

results = collection.query(
    query_texts=["I really want a vacuum cleaner"],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)  
results

# COMMAND ----------

files = dbutils.fs.ls("/tmp/product_vdb")
for file in files:
    print(file.name)


# COMMAND ----------


# Create a new client with the same settings
client = chromadb.PersistentClient(path=persist_directory_ephemeral)

# Load the collection
collections = client.list_collections()
collections

# COMMAND ----------

# Load the collection
collection = client.get_collection(collections[0].name)

# COMMAND ----------

results = collection.query(
    query_texts=["I really want a vacuum cleaner"],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)  
results

# COMMAND ----------

#Sample Question
payload_pd = pd.DataFrame([["ipod"]],columns=['text'])
input_example = payload_pd
input_example

# COMMAND ----------

  #Testing the prediction function
  def predict(model_input):
    import json
    question = model_input.iloc[:,0].to_list() # get the first column
    results = collection.query(
    query_texts=question,
    n_results=3,
    )
    # The vector search is over the similar stack overflow questions. What needs to be concatenated are responses to the top 2 queries stored as metadata
    responses = {"table": results['ids'][0][:], 'description': results['documents'][0][:], 'schema':results['metadatas'][0][:]}
    result = {'Response': responses}
    return json.dumps(result)

# COMMAND ----------

predict(input_example)

# COMMAND ----------

import mlflow.pyfunc

class RecommendationFinder(mlflow.pyfunc.PythonModel):

  def load_context(self, context):
    import chromadb
    # Create a new client with the same settings
    self.client = chromadb.PersistentClient(path=context.artifacts["persist_directory"])
    # Load the collection
    self.collection = self.client.get_or_create_collection(name="product_reviews")


  def predict(self, context, model_input):
    import json
    question = model_input.iloc[:,0].to_list() # get the first column
    results = self.collection.query(
    query_texts=question,
    n_results=3,
    )
    # The vector search is over the similar stack overflow questions. What needs to be concatenated are responses to the top 2 queries stored as metadata
    responses = {"table": results['ids'][0][:], 'description': results['documents'][0][:], 'schema':results['metadatas'][0][:]}
    result = {'Response': responses}
    return json.dumps(result)
     

# COMMAND ----------

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

# COMMAND ----------

conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      'python={}'.format(PYTHON_VERSION),
      'pip',
      {
        'pip': [
          'mlflow',
          'transformers',
          'pandas',
          'chromadb',
          'cloudpickle=={}'.format(cloudpickle.__version__),
          'torch'],
      },
    ],
    'name': 'prod_recsys'
}

# COMMAND ----------

mlflow_pyfunc_model_path = "prod_recsys"
artifacts = {
   "persist_directory": persist_directory_ephemeral
}

# COMMAND ----------

mlflow.set_experiment(f"/Users/avinash.sooriyarachchi@databricks.com/prod_recsys")

# COMMAND ----------

mlflow.pyfunc.log_model(artifact_path=mlflow_pyfunc_model_path, python_model=RecommendationFinder(),artifacts=artifacts, conda_env=conda_env, input_example = input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC Register and deploy the logged model as detailed here: https://docs.databricks.com/en/machine-learning/model-serving/create-manage-serving-endpoints.html
# MAGIC

# COMMAND ----------

