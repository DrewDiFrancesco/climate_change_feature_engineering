default_args = {
    's3_bucket': "s3://climatechangeprojectdrewdifrancesco",
    'spark_driver_memory': '3g',
    'spark_log_level': 'ERROR',
    'hadoop_version': '3.3.6',
    'hadoop_jar_file_name':'hadoop-aws-3.3.6.jar',
    'hadoop_aws_jar_url':f"https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.6/hadoop-aws-3.3.6.jar",
    'aws_java_sdk_jar_url': f"https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.12.543/aws-java-sdk-1.12.543.jar",
    'hadoop_home_path':'/home/ec2-user/SageMaker/climate_change_project/hadoop-3.3.6',
#     'java_home_path': "/usr/lib/jvm/java-1.8.0-openjdk",
    'java_home_path':'/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.382.b05-1.amzn2.0.1.x86_64/jre',
    'aws_java_sdk_jar_file_name': 'aws-java-sdk-1.12.543.jar',
    'aws_java_sdk_kms_jar_file_name': 'aws-java-sdk-kms-1.12.543.jar',
    'aws_java_sdk_s3_jar_file_name': 'aws-java-sdk-s3-1.12.543.jar',
    'spark_home': "/home/ec2-user/anaconda3/lib/python3.10/site-packages/pyspark",
    'region':'us-east-2',
    'agg_script_folder': 'feature_engineering',
    'feature_data_folder': 'features'
}