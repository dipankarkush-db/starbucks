# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC http://go/ai_parse

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Demo
# MAGIC
# MAGIC ## Parse PDF using `ai_parse_document`
# MAGIC
# MAGIC **How to run:**
# MAGIC
# MAGIC 1. Clone this notebook
# MAGIC 2. Create a cluster >= 16.4
# MAGIC     - For Serverless GC, make sure it is 16.4+ and "Environment version: 2"
# MAGIC 3. Run it!
# MAGIC
# MAGIC **More info:**
# MAGIC - [Preview Doc](https://docs.google.com/document/d/1YkYyVbpKV1Q2dulzU9JqLYkrBcNeVF5x0gNw5GjL2FM/edit?tab=t.0)
# MAGIC - [Wiki](http://go/docparse)
# MAGIC - Questions? Join [#unstructured-to-structured](https://databricks.enterprise.slack.com/archives/C07HZAJURBM) slack channel

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Simple Exmple: `variant` output
# MAGIC
# MAGIC The `ai_parse_document` function output is in a `variant` type column.

# COMMAND ----------

%sh apt-get update && apt-get install -y poppler-utils
pip install --upgrade pip
pip install pdf2image

# COMMAND ----------

from pdf2image import convert_from_path
import matplotlib.pyplot as plt

pdf_path = "/Volumes/dkushari_uc/starbucks_genie/pdfs/ai-parse/public_tax_bucket_2t_wt_0.pdf"
images = convert_from_path(pdf_path, dpi=300)
plt.figure(figsize=(10, 10))
plt.imshow(images[0])
plt.axis("off")
plt.show()

# COMMAND ----------

# DBTITLE 1,SQL
# MAGIC %sql
# MAGIC SELECT
# MAGIC   path,
# MAGIC   ai_parse_document(content) AS parsed
# MAGIC FROM
# MAGIC   READ_FILES('/Volumes/dkushari_uc/starbucks_genie/pdfs/ai-parse/public_tax_bucket_2t_wt_0.pdf', format => 'binaryFile');

# COMMAND ----------

# DBTITLE 1,python
from pyspark.sql.functions import *

df = spark.read.format("binaryFile") \
  .load("/Volumes/dkushari_uc/starbucks_genie/pdfs/ai-parse/public_tax_bucket_2t_wt_0.pdf") \
  .withColumn(
    "parsed",
    ai_parse_document("content"))
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Advanced Exmple: access the `variant` output fields
# MAGIC
# MAGIC The `ai_parse_document` function output can be accessed (`document`, `error_status`, `corrupted_date`, `metadata`).

# COMMAND ----------

# DBTITLE 1,sql
# MAGIC %sql
# MAGIC WITH corpus AS (
# MAGIC   SELECT
# MAGIC     path,
# MAGIC     ai_parse_document(content) AS parsed
# MAGIC   FROM
# MAGIC     READ_FILES('/Volumes/dkushari_uc/starbucks_genie/pdfs/ai-parse/public_tax_bucket_2t_wt_0.pdf', format => 'binaryFile')
# MAGIC )
# MAGIC SELECT
# MAGIC   path,
# MAGIC   parsed:document:pages,
# MAGIC   parsed:document:elements,
# MAGIC   parsed:corrupted_data,
# MAGIC   parsed:error_status,
# MAGIC   parsed:metadata
# MAGIC FROM corpus;

# COMMAND ----------

# DBTITLE 1,python
from pyspark.sql.functions import *

df = (
  spark.read.format("binaryFile")
    .load("/Volumes/dkushari_uc/starbucks_genie/pdfs/ai-parse/public_tax_bucket_2t_wt_0.pdf")
    .withColumn("parsed", ai_parse_document(col("content")))
    .select(
      "path",
      expr("parsed:document:pages"),
      expr("parsed:document:elements"),
      expr("parsed:error_status"),
      expr("parsed:metadata")
    )
)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Advanced Exmple: cast to struct and .*
# MAGIC
# MAGIC The `ai_parse_document` function output can be casted to struct type (`document`, `error_status`, `metadata`).

# COMMAND ----------

# DBTITLE 1,SQL
# MAGIC %sql
# MAGIC WITH parsed_docs AS (
# MAGIC   SELECT
# MAGIC     path,
# MAGIC     CAST(
# MAGIC       ai_parse_document(content) AS STRUCT<
# MAGIC         document STRUCT<
# MAGIC           pages ARRAY<STRUCT<id INT, image_uri STRING>>,
# MAGIC           elements ARRAY<
# MAGIC             STRUCT<
# MAGIC               id INT,
# MAGIC               type STRING,
# MAGIC               content STRING,
# MAGIC               bbox ARRAY<STRUCT<coord ARRAY<INT>, page_id STRING>>,
# MAGIC               description STRING
# MAGIC             >
# MAGIC           >
# MAGIC         >,
# MAGIC         error_status ARRAY<STRUCT<error_message STRING, page_id INT>>,
# MAGIC         metadata STRUCT<id STRING, version STRING>
# MAGIC       >
# MAGIC     ) AS parsed
# MAGIC   FROM
# MAGIC     READ_FILES(
# MAGIC       '/Volumes/dkushari_uc/starbucks_genie/pdfs/ai-parse/public_tax_bucket_2t_wt_0.pdf',
# MAGIC       format => 'binaryFile'
# MAGIC     )
# MAGIC )
# MAGIC SELECT
# MAGIC   path,
# MAGIC   parsed.*
# MAGIC FROM
# MAGIC   parsed_docs;

# COMMAND ----------

# DBTITLE 1,python
from pyspark.sql.functions import *
from pyspark.sql.types import *

schema = StructType([
  StructField("document", StructType([
    StructField("pages", ArrayType(StructType([
      StructField("id", IntegerType()),
      StructField("image_uri", StringType())]))),
    StructField("elements", ArrayType(StructType([
      StructField("id", IntegerType()),
      StructField("type", StringType()),
      StructField("content", StringType()),
      StructField("bounding_box", ArrayType(StructType([
        StructField("coord", ArrayType(IntegerType())),
        StructField("page_id", IntegerType())
      ]))),
      StructField("description", StringType())])))])),
  StructField("error_status", StructType([
    StructField("error_message", StringType()),
    StructField("page_id", IntegerType())])),
  StructField("metadata", StructType([
    StructField("id", StringType()),
    StructField("version", StringType())]))
])

df = spark.read.format("binaryFile") \
  .load("/Volumes/dkushari_uc/starbucks_genie/pdfs/ai-parse/public_tax_bucket_2t_wt_0.pdf") \
  .withColumn(
    "parsed",
    ai_parse_document("content").cast(schema)) \
  .select(
    "path",
    "parsed.*")
display(df)

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH parsed_docs AS (
# MAGIC   SELECT
# MAGIC     path,
# MAGIC     CAST(
# MAGIC       ai_parse_document(content) AS STRUCT<
# MAGIC         document STRUCT<
# MAGIC           pages ARRAY<STRUCT<id INT, image_uri STRING>>,
# MAGIC           elements ARRAY<
# MAGIC             STRUCT<
# MAGIC               id INT,
# MAGIC               type STRING,
# MAGIC               content STRING,
# MAGIC               bbox ARRAY<
# MAGIC                 STRUCT<
# MAGIC                   coord ARRAY<INT>,
# MAGIC                   page_id STRING
# MAGIC                 >
# MAGIC               >,
# MAGIC               description STRING
# MAGIC             >
# MAGIC           >
# MAGIC         >,
# MAGIC         error_status ARRAY<STRUCT<error_message STRING, page_id INT>>,
# MAGIC         metadata STRUCT<id STRING, version STRING>
# MAGIC       >
# MAGIC     ) AS parsed
# MAGIC   FROM READ_FILES(
# MAGIC       '/Volumes/dkushari_uc/starbucks_genie/pdfs/ai-parse/public_tax_bucket_2t_wt_0.pdf',
# MAGIC       format => 'binaryFile'
# MAGIC   )
# MAGIC )
# MAGIC
# MAGIC -- FLATTEN EVERYTHING
# MAGIC SELECT
# MAGIC     path,
# MAGIC
# MAGIC     /* metadata */
# MAGIC     parsed.metadata.id                      AS metadata_id,
# MAGIC     parsed.metadata.version                 AS metadata_version,
# MAGIC
# MAGIC     /* page info */
# MAGIC     page.id                                 AS page_id,
# MAGIC     page.image_uri                          AS page_image_uri,
# MAGIC
# MAGIC     /* element info */
# MAGIC     elem.id                                 AS element_id,
# MAGIC     elem.type                               AS element_type,
# MAGIC     elem.content                            AS element_content,
# MAGIC     elem.description                        AS element_description,
# MAGIC
# MAGIC     /* bounding box */
# MAGIC     bbox.page_id                            AS bbox_page_id,
# MAGIC     bbox.coord                              AS bbox_coord,
# MAGIC     --coord_value                             AS bbox_coord_value
# MAGIC     bbox.coord                              AS bbox_coord_value
# MAGIC
# MAGIC FROM parsed_docs
# MAGIC
# MAGIC -- explode pages
# MAGIC LATERAL VIEW OUTER explode(parsed.document.pages) AS page
# MAGIC
# MAGIC -- explode elements
# MAGIC LATERAL VIEW OUTER explode(parsed.document.elements) AS elem
# MAGIC
# MAGIC -- explode bbox array
# MAGIC LATERAL VIEW OUTER explode(elem.bbox) AS bbox;
# MAGIC
