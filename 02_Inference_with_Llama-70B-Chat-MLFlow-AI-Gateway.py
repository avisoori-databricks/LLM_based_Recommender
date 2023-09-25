# Databricks notebook source
# MAGIC %md
# MAGIC Detailed information and step by step instructions on setting up a MosaicML Inference Endpoint for LlaMA-70B-Chat as a route in MLFlow AI Gateway is provided at https://www.databricks.com/blog/using-ai-gateway-llama2-rag-apps. 
# MAGIC
# MAGIC MLFlow AI Gateway documentation is given here: https://mlflow.org/docs/latest/gateway/index.html
# MAGIC
# MAGIC MosaicML Starter Tier details and instructions are provided here: https://www.mosaicml.com/blog/llama2-inference

# COMMAND ----------

# MAGIC %md
# MAGIC LLaMA Chat models have been instruction finetuned on data with a specific formatas seen below:
# MAGIC
# MAGIC ```
# MAGIC [INST] <<SYS>>
# MAGIC {{ system_prompt }}
# MAGIC <</SYS>>
# MAGIC
# MAGIC {{ user_message }} [/INST]
# MAGIC ```
# MAGIC
# MAGIC This is adhered to below. The system prompt has been styled to yield responses relevant to the context, i.e. that of an ecommerce website where the model is playing the role of the recommendataion system. The user mesage will detail the sequence of past purchases, the task to be performed and the disired nature of the output.

# COMMAND ----------

prompt = """[INST] <<SYS>>
You are an AI assistant functioning as a recommendation system for an ecommerce website. Keep your answers short and concise.
<</SYS>>
A user bought a pen, pencil, and a glue stick in that order. What item would he/she be likely to purchase next? Express as a python list 'next_item' [/INST]"""

# COMMAND ----------

# MAGIC %md
# MAGIC This can be generalized as follows

# COMMAND ----------

def format_string(system_prompt: str, items: str) -> str:
    formatted_string = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n"
    
    items_purchased = ', '.join(items[:-1]) + ', and ' + items[-1] if len(items) > 1 else items[0]
    
    next_item_question = f"A user bought {items_purchased} in that order. What single item would he/she be likely to purchase next?"
    
    formatted_string += f"{next_item_question} Express as a python list 'next_item' [/INST]"
    
    return formatted_string

# Example usage:
system_prompt = "You are an AI assistant functioning as a recommendation system for an ecommerce website. Be specific and limit your answers to the requested format. Do not provide an explanation"
items = ['scarf', 'beanie', 'ear muffs', 'long socks']
formatted_text = format_string(system_prompt, items)
print(formatted_text)

# COMMAND ----------

import mlflow.gateway
mlflow.gateway.set_gateway_uri("databricks")

# COMMAND ----------

# Query the completions route using the mlflow client
print(
    mlflow.gateway.query(
        route="mosaicml-llama2-70b-completions",
        data={
            "prompt": formatted_text,
            'temperature':0
        },
    )
)

# COMMAND ----------

