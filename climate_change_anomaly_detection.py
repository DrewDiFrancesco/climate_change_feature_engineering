import pandas as pd
import threading
import os
import json
import etl_helpers as etl_helpers
import config as config
from conf_variables import default_args
from spark_session import SparkManager
import importlib
import configparser


# Main function
def main(override_args=None, spark=None, config_manager =None):
    print(f"HERE is the override args: {override_args}")

    if override_args:
        default_args.update(override_args)

    print(f"HERE is the new default args: {default_args}")
    if config_manager is None:
        config_manager = config.Config(default_args)
    
    running_locally = config_manager.args['running_locally']

    if spark is None:
        print(f"Getting Spark")
        spark = SparkManager(config_manager.args).get_spark()
        print("Done getting spark")

    if config_manager.args['s3_bucket'] != '':
        print(f"Updating data_path because s3_bucket is not an empty string...")
        config_manager.args['data_path'] = config_manager.args['s3_bucket']+'/data'

    data_path = config_manager.args['data_path']    
    climate_change_df = etl_helpers.read_data(path_to_files=data_path, file_name="climate_change_data", spark=spark, df_type="pandas")
    climate_change_df = climate_change_df[['Country name','Series name','Year','Region','Income group']]
    feature_data_folder = config_manager.args['feature_data_folder']
    agg_script_folder = config_manager.args['agg_script_folder']

    if "folder_names" not in config_manager.args:
        root_path = config_manager.args['root_path']
        agg_script_folder = config_manager.args['agg_script_folder']
        config_manager.args['folder_names'] = get_feature_folders(root_path,agg_script_folder)
        
    folder_names = config_manager.args['folder_names']

    final_climate_change_list = []

    for folder_name in folder_names:
        climate_change_df_temp = get_merge_feature_data(folder_name,root_path,feature_data_folder,climate_change_df,data_path,spark,agg_script_folder)
        final_climate_change_list.append(climate_change_df_temp)
    
    final_climate_change_df = pd.concat(final_climate_change_list, ignore_index=True)


    print(final_climate_change_df.columns)

def get_feature_folders(root_path,agg_script_folder):
    feature_dir = os.path.join(root_path, agg_script_folder)

    if not os.path.exists(feature_dir) or not os.path.isdir(feature_dir):
        print("The 'feature_engineering' directory does not exist or is not a directory.")
        return []

    folder_names = [name for name in os.listdir(feature_dir) if os.path.isdir(os.path.join(feature_dir, name))]
    return folder_names

def get_merge_feature_data(folder_name,root_path,feature_data_folder,df,data_path,spark,agg_script_folder):
    feature_data_path = f'{data_path}/{feature_data_folder}'
    feature_df = etl_helpers.read_data(path_to_files=feature_data_path, file_name=folder_name, spark=spark, df_type="pandas")
    config = configparser.ConfigParser()
    feature_directory = f'{root_path}/{agg_script_folder}/{folder_name}'
    config.read(f'{feature_directory}/{folder_name}_agg_config.ini')
    join_cols = config['base_args']['selected_cols']
    join_cols = [col.strip() for col in join_cols.strip('[]').split(',')]
    join_cols.remove("Value")
    df = df.merge(feature_df,on=join_cols)
    return df 

if __name__ == "__main__":
    root_path = os.path.dirname(os.path.abspath(__file__))
    drews_conf = {'data_path': '/Users/drewdifrancesco/Desktop/data',
                    'root_path': root_path,
                    's3_bucket': ''}
    main(drews_conf)