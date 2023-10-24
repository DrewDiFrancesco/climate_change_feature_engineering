import logging
import os
# import urllib.request
from pyspark.sql import SparkSession
# from pyspark import SparkConf, SparkContext
import botocore.session
# import sagemaker_pyspark
# import pyspark

class SparkManager:
    def __init__(self, args: dict, job_name=None):

        """
        Initializes a spark session object
        
        Args:
            args (dict): The configuration you want to pass to the object
            job_name (string): An optional name you can call the spark session.  Lets you know what you used the session for

        Attributes:
            self.spark (spark session)
            self.logger (logger variable)
            self.log_filepath (directory that holds the log files)
        """
        
        if not isinstance(args, dict):
            args = vars(args)
        self.args = args

        if not job_name:
            job_name = type(self).__name__
        
        self.job_name = job_name
#         self.args['running_locally'] = running_locally
        self.spark = None
        self.logger = None
        self.log_filepath = os.path.join(self.args['root_path'], "logs")
        if not os.path.exists(self.log_filepath):
            os.mkdir(self.log_filepath)

    def initialize_logger(self, log_filepath):
        
        """
        Sets the logger attribute equal to a logging variable 

        Args:
            log_filepath (string): directory that holds the log files
        """

        log_file = os.path.join(log_filepath, f"{self.job_name}.log")

        logging.basicConfig(
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            format='[%(asctime)s] %(module)s: %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        spark_logger = logging
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)

        spark_logger.info(f"Initialized logger to write to log file {log_file}")

        self.logger = spark_logger

    def get_spark(self, jar_file_location=None):
        
        """
        Sets up a spark session dependent on if you are running locally or if you are using s3
        """

        print(f"checking to see if running locally...")
        if self.args['running_locally']:
            print(f"Setting up spark for local")
            spark_config = SparkSession.builder.master("local").appName(self.job_name) \
                .config("spark.master", "local[2]") \
                .config("spark.sql.debug.maxToStringFields", 2000) \
                .config("spark.eventLog.dir", self.log_filepath) \
                .config("spark.eventLog.enabled", "true") \
                .config("spark.driver.memory", self.args['spark_driver_memory']) \
                .config("spark.executor.memory", self.args['spark_driver_memory'])
            if jar_file_location:
                spark_config.config("spark.jars", jar_file_location)
            spark_session = spark_config.getOrCreate()
            spark_session.sparkContext.setLogLevel(self.args['spark_log_level'])
            self.spark = spark_session

        else:
            print(f"Setting up Spark for sagemaker")
            root_path = self.args['root_path']
            region=self.args['region']
            hadoop_jar_file_name = self.args['hadoop_jar_file_name']
            hadoop_home_path = self.args['hadoop_home_path']
            aws_java_sdk_jar_file_name = self.args['aws_java_sdk_jar_file_name']
            hadoop_aws_jar_path = root_path + f"/{hadoop_jar_file_name}"
            aws_java_sdk_jar_url = root_path + f"/{aws_java_sdk_jar_file_name}"
            aws_java_sdk_kms_jar_file_name = self.args['aws_java_sdk_kms_jar_file_name']
            aws_java_sdk_kms_jar_url = root_path + f"/{aws_java_sdk_kms_jar_file_name}"
            aws_java_sdk_s3_jar_file_name = self.args['aws_java_sdk_s3_jar_file_name']
            aws_java_sdk_s3_jar_url = root_path + f"/{aws_java_sdk_s3_jar_file_name}"
            jar_files = [hadoop_aws_jar_path, aws_java_sdk_jar_url, aws_java_sdk_kms_jar_url, aws_java_sdk_s3_jar_url]
            jar_files = ",".join(jar_files)
            end_point = f"s3.{region}.amazonaws.com"
            access_key = self.args['access_key']
            secret_key = self.args['secret_key']
            
            print(f"Setting environment variables")
            os.environ['SPARK_HOME'] =self.args['spark_home']
            path = os.environ['PATH']
            os.environ['HADOOP_HOME'] = hadoop_home_path
            os.environ['HADOOP_CONF_DIR'] = f"{hadoop_home_path}/etc/hadoop"
            os.environ['PATH'] = f"{path}:{hadoop_home_path}/bin"
            os.environ['JAVA_HOME'] = self.args['java_home_path']
#             os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars {} pyspark-shell'.format(','.join([hadoop_aws_jar_path, aws_java_sdk_jar_url]))

    
#             All it took is to set `spark.hadoop.fs.s3.impl` to `org.apache.hadoop.fs.s3a.S3AFileSystem` and prefix the URI with s3a://
            
#             if 'sc' not in globals():
#                 conf = pyspark.SparkConf()
#                 sc = SparkContext(conf=conf)

            session = botocore.session.get_session()
            credentials = session.get_credentials()

            print(f"credentials.access_key: {credentials.access_key}")
            print(f"credentials.secret_key: {credentials.secret_key}")
            print(f"end_point: {end_point}")
            print(f"jar_files: {jar_files}")

            spark_config = SparkSession.builder.master("local[2]").appName(self.job_name)
            
            # New
            spark_config.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            spark_config.config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            spark_config.config("spark.jars.packages", 
                                "io.delta:delta-core_2.12:1.1.0,"
                                "org.apache.hadoop:hadoop-aws:3.2.2,"
                                "com.amazonaws:aws-java-sdk-bundle:1.12.180")
            # New
            spark_config.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            spark_config.config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            
            spark_config.config('spark.hadoop.fs.s3a.access.key', credentials.access_key)
            spark_config.config('spark.hadoop.fs.s3a.secret.key', credentials.secret_key)
            spark_config.config("spark.hadoop.fs.s3a.endpoint", end_point)
#             spark_config.config("spark.jars", jar_files)
            
#             spark_config.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

            spark_session = spark_config.getOrCreate()
    
            # New
            spark_session._jsc.hadoopConfiguration().set("com.amazonaws.services.s3.enableV4", "true")
            credentials_provider = "com.amazonaws.auth.InstanceProfileCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain" 
            spark_session._jsc.hadoopConfiguration().set("fs.s3a.aws.credentials.provider",credentials_provider)
            spark_session._jsc.hadoopConfiguration().set("fs.AbstractFileSystem.s3a.impl", "org.apache.hadoop.fs.s3a.S3A")


#             conf = SparkConf()
#                     .set("spark.driver.extraClassPath", ":".join(sagemaker_pyspark.classpath_jars()))
#             spark_session = (
#                 SparkSession
#                 .builder
#                 .config(conf=conf) \
#                 .config('fs.s3a.access.key', credentials.access_key)
#                 .config('fs.s3a.secret.key', credentials.secret_key)
#                 .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
#                 .config("spark.hadoop.fs.s3a.endpoint", end_point)
# #                 .config("spark.jars", ",".join([hadoop_aws_jar_path, aws_java_sdk_jar_url]))  # Use a list of JAR paths
#                 .appName(self.job_name)
#                 .getOrCreate()
#             )
            
            s3a_impl = spark_session.sparkContext.getConf().get("spark.hadoop.fs.s3a.impl")
            s3a_access_key = spark_session.sparkContext.getConf().get("spark.hadoop.fs.s3a.access.key")
            s3a_secret_key = spark_session.sparkContext.getConf().get("spark.hadoop.fs.s3a.secret.key")
            s3a_endpoint = spark_session.sparkContext.getConf().get("spark.hadoop.fs.s3a.endpoint")
            jars = spark_session.sparkContext.getConf().get("spark.jars")

            print(f"S3A FileSystem implementation: {s3a_impl}")
            print(f"S3A Access Key: {s3a_access_key}")
            print(f"S3A Secret Key: {s3a_secret_key}")
            print(f"S3A Endpoint: {s3a_endpoint}")
            print(f"JARs in classpath: {jars}")
            
#             hadoop_jar_file_name = self.args['hadoop_jar_file_name']
#             hadoop_aws_jar_url = self.args['hadoop_aws_jar_url']
# #             aws_java_sdk_jar_url = self.args['aws_java_sdk_jar_url']
# #             hadoop_version = self.args['hadoop_version']
#             print(f"hadoop_aws_jar_path: {hadoop_aws_jar_path}")
            
# #             print("Getting aws java sdk jar file...")
# #             urllib.request.urlretrieve(aws_java_sdk_jar_url, root_path + "/aws-java-sdk.jar")
# #             print(f"Getting hadoop aws jar file...")
# #             urllib.request.urlretrieve(hadoop_aws_jar_url, root_path + f"/hadoop-aws-{hadoop_version}.jar")
            
#             print(f"Building spark session...")
#             spark_config = SparkSession.builder.master("yarn").appName(self.job_name)
#             if jar_file_location:
#                 spark_config.config("spark.jars", jar_file_location)
#             spark_config.config("spark.driver.memory", self.args['spark_driver_memory']) \
#             spark_config.config("spark.executor.memory", self.args['spark_driver_memory'])
#             print(f"Setting configs for endpoint")
#             spark_config.config("spark.hadoop.fs.s3a.endpoint", end_point)
#             print(f"Setting configs for access key")
#             spark_config.config("spark.hadoop.fs.s3a.access.key", access_key)
#             print(f"Setting configs for secret key")
#             spark_config.config("spark.hadoop.fs.s3a.secret.key", secret_key)
#             spark_config.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
#             print(f"Setting configs for jar file")
#             spark_config.config("spark.jars", hadoop_aws_jar_path)
#             print(f"Creating spark session")
#             spark_session = spark_config.getOrCreate()
            print(f"Setting spark context log level...")
            spark_session.sparkContext.setLogLevel(self.args['spark_log_level'])
            self.spark = spark_session
            print(f"Finished creating spark session")

        return spark_session