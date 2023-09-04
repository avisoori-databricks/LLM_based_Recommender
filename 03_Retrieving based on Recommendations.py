# Databricks notebook source
# MAGIC %md
# MAGIC Detailed information and step by step instructions on setting up a MosaicML Inference Endpoint for LlaMA-70B-Chat as a route in MLFlow AI Gateway is provided at https://www.databricks.com/blog/using-ai-gateway-llama2-rag-apps. 
# MAGIC
# MAGIC MLFlow AI Gateway documentation is given here: https://mlflow.org/docs/latest/gateway/index.html
# MAGIC
# MAGIC MosaicML Starter Tier details and instructions are provided here: https://www.mosaicml.com/blog/llama2-inference

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

# COMMAND ----------

mlflow.gateway.set_gateway_uri("databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC As discussed earlier LLaMA Chat models have been instruction finetuned on data with a specific formatas seen below:
# MAGIC
# MAGIC ```
# MAGIC [INST] <<SYS>>
# MAGIC {{ system_prompt }}
# MAGIC <</SYS>>
# MAGIC
# MAGIC {{ user_message }} [/INST]
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC The prompt generation function for this recommendation task, created based on some prompt engineering while conforming to the above format, is as follows

# COMMAND ----------

def format_string(system_prompt, items):
    formatted_string = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n"
    
    items_purchased = ', '.join(items[:-1]) + ', and ' + items[-1] if len(items) > 1 else items[0]
    
    next_item_question = f"A user bought {items_purchased} in that order. What single item would he/she be likely to purchase next?"
    
    formatted_string += f"{next_item_question} Express as a python list 'next_item' with an inline description[/INST]"
    
    return formatted_string

# Example usage:
system_prompt = "You are an AI assistant functioning as a recommendation system for an ecommerce website. Be specific and limit your answers to the requested format."
#items = ['scarf', 'beanie', 'ear muffs', 'long socks']
items = ['pen', 'pencil', 'eraser']
formatted_text = format_string(system_prompt, items)
print(formatted_text)

# COMMAND ----------

# Query the completions route using the mlflow client

next_item = mlflow.gateway.query(
        route="mosaicml-llama2-70b-completions",
        data={
            "prompt": formatted_text,
            'temperature':0
        },)['candidates'][0]['text']

next_item

# COMMAND ----------

df = pd.DataFrame({'text': [next_item]})
display(df)

# COMMAND ----------

# # Set the token value
# token_value = "your-token-value-here"

# # Store the token in the Spark configuration
# spark.conf.set("com.databricks.training.apiToken", token_value)

# # You can now retrieve the token using the code you provided

# You can retrieve the PAT token using the code you provided
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

def vector_search(dataset):
  url = 'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/prod_recsys/invocations'
  headers = {'Authorization': f'Bearer {API_TOKEN}', 
'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')}
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

  return response.json()

# COMMAND ----------

recommendations = vector_search(df)
recommendations['predictions']

# COMMAND ----------

json.loads(recommendations['predictions'])['Response']['schema']

# COMMAND ----------

# MAGIC %md
# MAGIC The entire LLM based recommendation task can can be neatly wrapped into a function as follows

# COMMAND ----------

def LLM4rec(system_prompt, items):
  #This random item can be anything, to ensure there's something to recommend given an absolute cold start
  #i.e. the user has not purchased anything yet/ visited any item's webpage before
  starter_items =  [ "Smartphone", "Laptop", "Headphones", "T-shirt", "Jeans", "Running Shoes", "Blender", "Novel", "Board Game", "Wristwatch" ]
  if len(items)==0:
    items.append(random.sample(starter_items, 1))
  prompt = format_string(system_prompt, items)
  next_item = mlflow.gateway.query(
        route="mosaicml-llama2-70b-completions",
        data={
            "prompt": prompt,
            'temperature':0
        },)['candidates'][0]['text']
  candidate = pd.DataFrame({'text': [next_item]})
  results = vector_search(candidate)
  recommendations = {'recommendations': json.loads(results['predictions'])['Response']['schema']}
  return recommendations

# COMMAND ----------

# MAGIC %md
# MAGIC Testing the above function 

# COMMAND ----------

user_purchase_histories = {'Ben':['hammer', 'nails', 'saw', 'super glue'],
                           'Jess': ['pen', 'pencil', 'eraser'],
                           'Kumar': ['bicycle'],
                           'Jose': ['car seat', 'diapers', 'tissue'],
                           'noob': []}

# COMMAND ----------

system_prompt = "You are an AI assistant functioning as a recommendation system for an ecommerce website. Be specific and limit your answers to the requested format."

# COMMAND ----------

LLM4rec(system_prompt, user_purchase_histories['Jess'])

# COMMAND ----------

# MAGIC %md
# MAGIC Great!! All these recommendations make sense. This is only meant to be an example, but there is an entire research field actively pursuing this idea

# COMMAND ----------

# Create a dropdown widget with the names from the dictionary keys
name_dropdown = widgets.Dropdown(
    options=user_purchase_histories.keys(),
    description='Select a name:',
)

# Create a button widget to trigger the recommendation
recommend_button = widgets.Button(description='Recommend')

# Create an output widget to display the recommendations
output = widgets.Output()

# COMMAND ----------

# Define a function to handle button click
def recommend_button_click(b):
    selected_name = name_dropdown.value
    recommendations = LLM4rec(system_prompt, user_purchase_histories[selected_name])
    
    # Display the recommendations
    with output:
        output.clear_output()
        print("Recommendations:")
        print(recommendations)
      

# COMMAND ----------

# Connect the button click event to the function
recommend_button.on_click(lambda b: recommend_button_click(b))

# Display the widgets
display(name_dropdown, recommend_button, output)