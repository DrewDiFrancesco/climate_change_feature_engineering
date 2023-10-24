#!/usr/bin/env python3
import pandas as pd

def read_data(path_to_files="", file_name="", spark=None, df_type="spark"):
    """
    This function reads and returns saved spark dataframes

    Args:
        path_to_files (string): The directory location of where the dataframe you want to read is saved
        file_name (string): The name of the saved dataframe
        spark: The variable that represents the spark session
        df_type: "spark" if you want a spark df and anything else if you want a pandas df
    Returns:
        df: returns a spark dataframe or a pandas dataframe
    """

    data_path = f"{path_to_files}/{file_name}".rstrip('/')
    
    if df_type == "spark":
        df = spark.read.parquet(data_path)
    else:
        df = pd.read_parquet(data_path)

    return df

def write_data(df, path_to_files="", file_name="", mode="overwrite", partition_columns=None):
    """
    This function saves a spark dataframe to the given path directory, partitioning the data if given columns to partition on
    
    Args:
        df (spark df): The spark dataframe you want to save
        path_to_files (string): The directory location where you want to save the data
        file_name (string): The name you want the dataframe to be saved as
        mode (string): Can be "overwrite" or "append", depending on whether you wish to save a completely new dataframe or add to an existing one
        partition_columns (list): A list of the columns you wish to partition on
    """
    data_path = f"{path_to_files}/{file_name}".rstrip('/')
    print(f"Saving to: {data_path}")
    if partition_columns:
        df.write.partitionBy(*partition_columns).mode(mode).parquet(data_path)
    else:
        df.write.mode(mode).parquet(data_path)
        
        