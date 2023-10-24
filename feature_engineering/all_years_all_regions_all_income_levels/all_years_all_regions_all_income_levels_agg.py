import configparser 
from pyspark.sql import functions as spark_functions 
from pyspark.sql.window import Window 
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.recommendation import ALS
import datetime
import etl_helpers as etl_helpers
import os
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.recommendation import ALS
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.linalg import DenseVector
from pyspark.sql.functions import udf

config = configparser.ConfigParser()
current_directory = os.path.dirname(os.path.abspath(__file__))

script_path = os.path.abspath(__file__)
script_name = os.path.basename(script_path).replace('.py','')

config.read(f'{current_directory}/{script_name}_config.ini')

aggregation_name = config['base_args']['aggregation_name']
selected_cols_str = config['base_args']['selected_cols']
selected_cols = [col.strip() for col in selected_cols_str.strip('[]').split(',')]
peer_group_str = config['base_args']['peer_group']
peer_group = [col.strip() for col in peer_group_str.strip('[]').split(',')]
time_span = config['base_args']['time_span']


def calculate_risk_score(df, selected_column, window_spec):

    risk_score_column_name = f"{selected_column}_risk_score"

    # Calculate the mean and standard deviation of the features
    df = df.withColumn("mean", spark_functions.mean(spark_functions.col(selected_column)).over(window_spec))
    df = df.withColumn("stddev", spark_functions.stddev(spark_functions.col(selected_column)).over(window_spec))

    # Calculate the z-score for each feature
    df = df.withColumn("z_score", (spark_functions.col(selected_column) - spark_functions.col("mean")) / spark_functions.col("stddev"))

    # Calculate the risk score based on the absolute z-score
    df = df.withColumn(risk_score_column_name, spark_functions.sum(spark_functions.abs(spark_functions.col("z_score"))).over(window_spec))

    return df

# def calculate_risk_score(df, columns_to_use):
#     # Assemble the feature vectors
#     assembler = VectorAssembler(inputCols=columns_to_use, outputCol="features")
#     df_assembled = assembler.transform(df)

#     # Create and train an autoencoder model
#     autoencoder = ALS(rank=10, maxIter=10, regParam=0.01, userCol="features", itemCol="features")

#     # Fit the model
#     model = autoencoder.fit(df_assembled)

#     # Get the transformed data (reconstructed features)
#     transformed_data = model.transform(df_assembled)

#     # Calculate a risk score (e.g., reconstruction error)
#     reconstruction_error = transformed_data.select(
#         sum(abs(col("features") - col("prediction"))).alias("risk_score")
#     )

#     # Add the risk score back to the original DataFrame
#     df_assembled = df_assembled.join(reconstruction_error, lit(1))
#     print(f"Done getting risk score...")

#     return df_assembled

def main(df, data_path):
    initial_columns = df.columns

    data_path = f"{data_path}/features"
    
    final_columns = selected_cols + ['peer_value','peer_count', 'percent_difference', 'Value_risk_score', 'percent_difference_risk_score']
    final_columns.remove('Value')

    if time_span != 0:
        if time_span == 1:
            df = df.withColumn("time_span", df.Year.cast("string"))
        else:
            df = df.withColumn("time_span", spark_functions.lit(time_span))
            df = df.withColumn("end_year", df["year"].cast("int") + df["time_span"] - 1)
            df = df.withColumn("time_span", spark_functions.concat(df["year"], spark_functions.lit("-"), df["end_year"]))
        peer_group.append("time_span")

    window_spec = Window.partitionBy(peer_group)

    df = df.withColumn("Value", df.Value.cast("double"))
    df = df.withColumn("avg_value", spark_functions.mean("Value").over(window_spec))
    df = df.withColumn("Value", spark_functions.when(df["value"].isNull(), df["avg_value"]).otherwise(df["value"]))   
    df = df.filter(df.Value.isNotNull()) 
    df = df.withColumn("peer_value", spark_functions.sum("Value").over(window_spec))
    df = df.withColumn("peer_value", df.peer_value - df.Value)
    df = df.withColumn("peer_count", spark_functions.collect_set('Country name').over(window_spec))
    df = df.withColumn("peer_count", spark_functions.size("peer_count"))
    df = df.withColumn("peer_count", df.peer_count - 1)
    df = df.withColumn("peer_value", df.peer_value / df.peer_count)
    df = df.withColumn("percent_difference", (df.Value - df.peer_value) / df.Value)
    df = df.withColumn("percent_difference", df.percent_difference.cast("double"))
    df = calculate_risk_score(df, "percent_difference", window_spec)
    df = calculate_risk_score(df, "Value", window_spec)   


    # df = df.withColumn("aggregation_name", spark_functions.lit(aggregation_name))
    df = df.select(final_columns)

    for column in final_columns:
        if column not in initial_columns:
            df = df.withColumnRenamed(column,f"{column}_{aggregation_name}")

    etl_helpers.write_data(df, path_to_files=data_path, file_name=aggregation_name)

