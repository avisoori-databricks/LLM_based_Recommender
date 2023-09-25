# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch-preview
# MAGIC dbutils.library.restartPython()

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

# MAGIC %md
# MAGIC ## Redisplaying code for sequential recommendations from Notebook 5

# COMMAND ----------

import re
import mlflow.gateway # Be sure to use Databricks MLR 14 or later
import os
import requests
import numpy as np
import pandas as pd
import json
import random
from ipywidgets import interact, widgets
from IPython.display import display
import asyncio
from databricks.vector_search.client import VectorSearchClient
from typing import List, Dict, Any, Union

# COMMAND ----------

def vector_search(client: 'databricks.vector_search.client.VectorSearchClient', num_items: int, next_item: str)->List[str]:
  result = client.similarity_search(
    index_name = "vs_catalog.vs_schema.instacart-grocery-product-index",
    query_text = next_item,
    columns = ["id", "product"], # columns to return
    num_results = num_items)
  return [item[-2] for item in result['result']['data_array']]

# COMMAND ----------

def format_string(system_prompt: str, items: List[str])-> str:
    formatted_string = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n"
    
    items_purchased = ', '.join(items[:-1]) + ', and ' + items[-1] if len(items) > 1 else items[0]
    
    next_item_question = f"A user bought {items_purchased} in that order. What single item would he/she be likely to purchase next?"
    
    formatted_string += f"{next_item_question} Express as a python list 'next_item' with an inline description[/INST]"
    
    return formatted_string

# Example usage:
system_prompt = "You are an AI assistant functioning as a recommendation system for an ecommerce website. Be specific and limit your answers to the requested format."
#items = ['scarf', 'beanie', 'ear muffs', 'long socks']
items = ['chicken', 'eggs', 'olive oil']
formatted_text = format_string(system_prompt, items)
print(formatted_text)

# COMMAND ----------

def LLM4rec(system_prompt: str, items: List[str], client: 'databricks.vector_search.client.VectorSearchClient', num_recs: int)-> Dict[str, List[str]]:
  #This random item can be anything, to ensure there's something to recommend given an absolute cold start
  #i.e. the user has not purchased anything yet/ visited any item's webpage before
  starter_items =  ['chicken', 'eggs', 'olive oil', 'pastrami', 'spinach', 'lettuce', 'quinoa']
  if len(items)==0:
    items.append(random.sample(starter_items, 1))
  prompt = format_string(system_prompt, items)
  next_item = mlflow.gateway.query(
        route="mosaicml-llama2-70b-completions",
        data={
            "prompt": prompt,
            'temperature':0.5 #0
        },)['candidates'][0]['text']
  recommendations = vector_search(client, num_recs,next_item)
  result = {'recommendations': recommendations}
  return result

# COMMAND ----------

system_prompt = "You are an AI assistant functioning as a recommendation system for an ecommerce website. Be specific and limit your answers to the requested format."

# COMMAND ----------

user_purchase_histories = {'Ben':['hammer', 'nails', 'saw', 'super glue'],
                           'Jess': ['pen', 'pencil', 'eraser'],
                           'Kumar': ['bicycle'],
                           'Jose': ['car seat', 'diapers', 'tissue'],
                           'noob': []}

# COMMAND ----------

system_prompt = "You are an AI assistant functioning as a recommendation system for an ecommerce website. Be specific and limit your answers to the requested format."

# COMMAND ----------

LLM4rec(system_prompt, user_purchase_histories['Ben'], client, 2)

# COMMAND ----------

def sequentialrec(system_prompt: str, items: List[str], client: 'databricks.vector_search.client.VectorSearchClient', num_recs: int)->List[str]:
  history = items
  new_recs = []
  for i in range(num_recs):
    new_recs = LLM4rec(system_prompt, history, client, 1)['recommendations']
    history += new_recs

  return history

# COMMAND ----------

#An example sequential recommendation:
items = ["Sea Salt Garden Veggie Chips", "Atlantic Salmon Fillet", "Salsa Casera Mild"]
sequentialrec(system_prompt, items, client, 5)

# COMMAND ----------

df = spark.sql("SELECT * FROM recommender_evaluation_table").toPandas()
display(df)

# COMMAND ----------

prev_purchases = [list(arr) for arr in df.previous_purchases]
prev_purchases[:3]

# COMMAND ----------

recommendations = [sequentialrec(system_prompt,item, client, 5)[-5:] for item in prev_purchases]
recommendations[:3]

# COMMAND ----------

ground_truth = [list(arr) for arr in df.ground_truth]
ground_truth[:3]

# COMMAND ----------

data = {'Ground_Truth':ground_truth,
        'Recommendations' : recommendations}
df = pd.DataFrame(data)

# COMMAND ----------

def convert_to_binary_relevance(row):
    return [1 if rec in row['Ground_Truth'] else 0 for rec in row['Recommendations']]

df['Binary_Relevance'] = df.apply(convert_to_binary_relevance, axis=1)

# COMMAND ----------

def precision_at_k(r, k):
    assert k >= 1
    r = r[:k]
    num_relevant = sum(r)
    return num_relevant / k


# COMMAND ----------

def average_precision(r):
    precisions = [precision_at_k(r, k + 1) for k, rel in enumerate(r) if rel]
    if not precisions:
        return 0
    return sum(precisions) / len(precisions)


# COMMAND ----------

def map_at_k(df, k):
    aps = df['Binary_Relevance'].apply(lambda r: average_precision(r[:k]))
    return aps.mean()


# COMMAND ----------

k = 5
print("MAP@{}: {:.2f}".format(k, map_at_k(df, k)))


# COMMAND ----------

k, map_at_k(df, k)

# COMMAND ----------

#k, map_at_k(df, k) at 3 was 0 as well