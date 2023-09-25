# Databricks notebook source
# MAGIC %sql
# MAGIC -- CREATE A CATALOG, SCHEMA AND VOLUME TO STORE DATA NEEDED FOR THIS. IN PRACTICE, YOU COULD USE AN EXISTING VOLUME
# MAGIC CREATE CATALOG IF NOT EXISTS llm_recommender;
# MAGIC USE CATALOG llm_recommender;
# MAGIC CREATE DATABASE IF NOT EXISTS purchase_data;
# MAGIC USE DATABASE purchase_data;
# MAGIC CREATE VOLUME IF NOT EXISTS purchase_raw;

# COMMAND ----------

import pandas as pd
import random
from pyspark.sql.types import *
import pyspark.sql.functions as fn
from pyspark.sql import window as w
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
from typing import List

# COMMAND ----------

#Define creation helper function
def read_data(file_path, schema):
  df = (
    spark
      .read
      .csv(
        file_path,
        header=True,
        schema=schema
        )
    )
  return df
 
def write_data(df, table_name):
   _ = (
       df
        .write
        .format('delta')
        .mode('overwrite')
        .option('overwriteSchema','true')
        .saveAsTable(table_name)
       )  

# COMMAND ----------

#Concat order_products_prior and train and save in the same volume
prior = pd.read_csv('/Volumes/llm_recommender/purchase_data/purchase_raw/order_products__prior.csv')
train= pd.read_csv('/Volumes/llm_recommender/purchase_data/purchase_raw/order_products__train.csv')
concatted = pd.concat([prior, train])
concatted.to_csv('/Volumes/llm_recommender/purchase_data/purchase_raw/order_products.csv')
concatted = pd.read_csv('/Volumes/llm_recommender/purchase_data/purchase_raw/order_products.csv')
display(concatted)

# COMMAND ----------

dbutils.fs.ls('/Volumes/llm_recommender/purchase_data/purchase_raw')

# COMMAND ----------

#paths of raw .csv files in llm_recommender volume
config = {'aisles_path': '/Volumes/llm_recommender/purchase_data/purchase_raw/aisles.csv',
         'departments_path': '/Volumes/llm_recommender/purchase_data/purchase_raw/departments.csv',
         'order_products_path': '/Volumes/llm_recommender/purchase_data/purchase_raw/order_products.csv',
         'orders_path': '/Volumes/llm_recommender/purchase_data/purchase_raw/orders.csv',
         'products_path' : '/Volumes/llm_recommender/purchase_data/purchase_raw/products.csv', 
         'database': 'movie_data'
         }

# COMMAND ----------

#Load the Data to Tables
# orders data
# ---------------------------------------------------------
orders_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('user_id', IntegerType()),
  StructField('eval_set', StringType()),
  StructField('order_number', IntegerType()),
  StructField('order_dow', IntegerType()),
  StructField('order_hour_of_day', IntegerType()),
  StructField('days_since_prior_order', FloatType())
  ])
 
orders = read_data(config['orders_path'], orders_schema)
write_data( orders, '{0}.orders'.format(config['database']))
# ---------------------------------------------------------
 
# products
# ---------------------------------------------------------
products_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('product_name', StringType()),
  StructField('aisle_id', IntegerType()),
  StructField('department_id', IntegerType())
  ])
 
products = read_data( config['products_path'], products_schema)
write_data( products, '{0}.products'.format(config['database']))
# ---------------------------------------------------------
 
# order products
# ---------------------------------------------------------
order_products_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('product_id', IntegerType()),
  StructField('add_to_cart_order', IntegerType()),
  StructField('reordered', IntegerType())
  ])
 
order_products = read_data( config['order_products_path'], order_products_schema)
write_data( order_products, '{0}.order_products'.format(config['database']))
# ---------------------------------------------------------
 
# departments
# ---------------------------------------------------------
departments_schema = StructType([
  StructField('department_id', IntegerType()),
  StructField('department', StringType())  
  ])
 
departments = read_data( config['departments_path'], departments_schema)
write_data( departments, '{0}.departments'.format(config['database']))
# ---------------------------------------------------------
 
# aisles
# ---------------------------------------------------------
aisles_schema = StructType([
  StructField('aisle_id', IntegerType()),
  StructField('aisle', StringType())  
  ])
 
aisles = read_data( config['aisles_path'], aisles_schema)
write_data( aisles, '{0}.aisles'.format(config['database']))
# ---------------------------------------------------------

# COMMAND ----------

display(products)

# COMMAND ----------

#We will need this later
# Assuming df is your DataFrame with columns 'product_id' and 'product_name'
products_list = products.select('product_id', 'product_name').rdd.map(lambda row: (row.product_id, row.product_name)).collect()

products_dict = dict(products_list)

print(products_dict)


# COMMAND ----------

#Present Tables in Database
display(
  spark
    .sql('SHOW TABLES')
  )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step 2: Generate Ratings
# MAGIC The records that make up the Instacart dataset represent grocery purchases. As would be expected in a grocery scenario, there are no explicit ratings provided in this dataset. Explicit ratings are typically found in scenarios where users are significantly invested (either monetarily or in terms of time or social standing) in the items they are purchasing or consuming. When we are considering apples and bananas purchased to have around the house as a snack or to be dropped in a kid's lunch, most users are just not interested in providing 1 to 5 star ratings on those items.
# MAGIC
# MAGIC We therefore need to examine the data for implied ratings (preferences). In a grocery scenario where items are purchased for consumption, repeat purchases may provide a strong signal of preference. Douglas Oard and Jinmook Kim provide a nice discussion of the various ways we might derive implicit ratings in a variety of scenarios and it is certainly worth considering alternative ways of deriving an input metric. However, for the sake of simplicity, we'll leverage the percentage of purchases involving a particular item as our implied rating:

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP VIEW IF EXISTS user_product_purchases;
# MAGIC  
# MAGIC CREATE VIEW user_product_purchases
# MAGIC AS
# MAGIC   SELECT
# MAGIC     monotonically_increasing_id() as row_id,
# MAGIC     x.user_id,
# MAGIC     x.product_id,
# MAGIC     x.product_purchases / y.purchase_events as rating
# MAGIC   FROM (  -- product purchases
# MAGIC     SELECT
# MAGIC       a.user_id,
# MAGIC       b.product_id,
# MAGIC       COUNT(*) as product_purchases
# MAGIC     FROM orders a
# MAGIC     INNER JOIN order_products b
# MAGIC       ON a.order_id=b.order_id
# MAGIC     INNER JOIN products c
# MAGIC       ON b.product_id=c.product_id
# MAGIC     GROUP BY a.user_id, b.product_id
# MAGIC     ) x 
# MAGIC   INNER JOIN ( -- purchase events
# MAGIC     SELECT 
# MAGIC       user_id, 
# MAGIC       COUNT(DISTINCT order_id) as purchase_events 
# MAGIC     FROM orders 
# MAGIC     GROUP BY user_id
# MAGIC     ) y
# MAGIC     ON x.user_id=y.user_id
# MAGIC     ;
# MAGIC     
# MAGIC SELECT *
# MAGIC FROM user_product_purchases;

# COMMAND ----------

# retrieve all ratings
ratings = spark.table('user_product_purchases')
 
# retrieve sampling of ratings
ratings_sampled = ratings.sample(0.10).cache()
 
print(f'Total:\t{ratings.count()}')
print(f'Sample:\t{ratings_sampled.count()}')

# COMMAND ----------

#Generate Train-Test Split
ratings_train, ratings_test = ratings.randomSplit([0.8, 0.2], random.randrange(100000))
ratings_sampled_train, ratings_sampled_test = ratings.randomSplit([0.8, 0.2], random.randrange(100000))

# COMMAND ----------

#Get Actual Selections

actual_selections = (ratings_sampled_test
    .withColumn('selections', fn.expr("collect_list(product_id) over(partition by user_id order by rating desc)")) 
    .filter(fn.expr("size(selections)<=10")) 
    .groupBy('user_id') 
      .agg(
        fn.max('selections').alias('selections')
        ))
display(actual_selections)
    

# COMMAND ----------

#Restrict the calculation to a 100 users to only ping the llm API fewer times during the evaluation and only sample those users who selected more than 7 items

# Filter rows where the size of the selections list is greater than 7
filtered_users = actual_selections.filter(fn.size(actual_selections.selections) > 5)
# # Sample 100 rows without replacement
# users_100 = filtered_users.sample(withReplacement=False, fraction=100/filtered_users.count())

display(filtered_users)


# COMMAND ----------

@pandas_udf(ArrayType(StringType()), PandasUDFType.SCALAR)
def map_ids_to_names_pandas_udf(product_ids_series: pd.Series) -> pd.Series:
    return product_ids_series.apply(lambda ids: [products_dict.get(pid, '') for pid in ids])

# Use the Pandas UDF to create a new column
filtered_users_with_names = filtered_users.withColumn("product_names", map_ids_to_names_pandas_udf(users_100["selections"]))


# COMMAND ----------

display(filtered_users_with_names)

# COMMAND ----------

products_dict[11205]

# COMMAND ----------


# UDF to get the first three product names
@pandas_udf(ArrayType(StringType()), PandasUDFType.SCALAR)
def previous_purchases(product_names_series: pd.Series) -> pd.Series:
    return product_names_series.apply(lambda names: names[:3] if len(names) > 3 else names)

# UDF to get all product names except the first three
@pandas_udf(ArrayType(StringType()), PandasUDFType.SCALAR)
def ground_truth(product_names_series: pd.Series) -> pd.Series:
    return product_names_series.apply(lambda names: names[3:] if len(names) > 3 else [])

# Use the Pandas UDFs to create the new columns
filtered_users_with_names = (filtered_users_with_names
                        .withColumn("previous_purchases", previous_purchases(filtered_users_with_names["product_names"]))
                        .withColumn("ground_truth", ground_truth(filtered_users_with_names["product_names"])))

# Show the resulting DataFrame
display(filtered_users_with_names)


# COMMAND ----------

# %sql
# DROP TABLE eval_table

# COMMAND ----------

filtered_users_with_names.write.saveAsTable('recommender_evaluation_table')

# COMMAND ----------

#Next step, do prediction, vector search, add that item to the previous items list, then next and next one in an autoregressive manner. Store the results obtained each time from the vector search in a list

# COMMAND ----------

