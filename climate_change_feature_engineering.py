import pandas as pd
import threading
import os
import json
import etl_helpers as etl_helpers
import config as config
from conf_variables import default_args
from spark_session import SparkManager
import importlib


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
    
    agg_script_folder = config_manager.args['agg_script_folder']
    
    if "folder_names" not in config_manager.args:
        root_path = config_manager.args['root_path']
        feature_engineering_folder = config_manager.args['agg_script_folder']
        config_manager.args['folder_names'] = get_feature_folders(root_path,feature_engineering_folder)
        
    folder_names = config_manager.args['folder_names']

    data_path = config_manager.args['data_path']
    # Load your pandas DataFrame here
    world_bank_non_agg_climate_change_data = etl_helpers.read_data(path_to_files=data_path, file_name="climate_change_data", spark=spark, df_type="spark")
    world_bank_non_agg_climate_change_data.cache()

    run_feature_engineering_scripts(folder_names, world_bank_non_agg_climate_change_data, data_path, agg_script_folder)

def get_feature_folders(root_path,feature_engineering_folder):
    feature_dir = os.path.join(root_path, feature_engineering_folder)

    if not os.path.exists(feature_dir) or not os.path.isdir(feature_dir):
        print("The 'feature_engineering' directory does not exist or is not a directory.")
        return []

    folder_names = [name for name in os.listdir(feature_dir) if os.path.isdir(os.path.join(feature_dir, name))]
    return folder_names

def run_feature_engineering_scripts(folder_names, df, data_path, script_directory):
    for folder_name in folder_names:
        script_name = f"{folder_name}_agg.py"
        script_path = os.path.join(script_directory, folder_name, script_name)
        
        if os.path.exists(script_path):
            # Dynamically import the script module
            script_module = importlib.import_module(f"{script_directory}.{folder_name}.{script_name.replace('.py', '')}")

            # Check if the 'main' function exists in the module
            if hasattr(script_module, 'main') and callable(script_module.main):
                # Execute the 'main' function with the provided arguments
                script_module.main(df, data_path)
            else:
                print(f"Error: 'main' function not found in {script_path}")
        else:
            print(f"Error: {script_path} does not exist.")


if __name__ == "__main__":
    root_path = os.path.dirname(os.path.abspath(__file__))
    # folder_names = ['all_years_all_regions_all_income_levels','']
    drews_conf = {'data_path': '/Users/drewdifrancesco/Desktop/data',
                    'root_path': root_path,
                    's3_bucket': ''}
    main(drews_conf)