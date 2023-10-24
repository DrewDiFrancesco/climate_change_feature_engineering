import os
import csv
from conf_variables import default_args
from argparse import Namespace



class Config:

    def __init__(self, args: dict = None, config_path: str = None) -> dict:

        """
        Initialize the configuration manager with provided arguments or defaults.

        This method initializes the configuration manager with a dictionary of arguments (`args`) and an optional
        configuration file path (`config_path`). If `args` are not provided, default arguments are obtained. The
        `running_locally` flag is set based on the environment. If a `config_path` is provided and the file has
        a `.env` extension, it reads and extracts the S3 bucket information and updates the arguments accordingly.

        Args:
            args (dict, optional): A dictionary of configuration arguments. If not provided, default arguments are used.
            config_path (str, optional): Path to a configuration file. If provided, it must have a `.env` extension.

        Returns:
            dict: The updated configuration arguments.

        Note:
            - If `args` are not provided, default arguments are obtained using the `get_default_args` method.
            - The `running_locally` flag is set based on the environment using the `is_running_locally` method.
            - If `config_path` is provided and the file has a `.env` extension, it is read to extract the S3 bucket
            information and update the arguments accordingly.
        """
        
        if not args:
            args = self.get_default_args()

        self.args = args
        self.args['running_locally'] = self.is_running_locally()

        if config_path:
            if os.path.splittext(config_path)[1].lower() == '.env':
                with open(config_path, 'r') as fIN:
                    reader = csv.reader(fIN)
                    for row in reader:
                        if row[0].split('=')[0] == 'S3_ENG_BUCKET':
                            bucket = row[0].split('=')[1]
                            args['bucket'] = bucket

    def get_default_args(self):
        
        """
        Retrieve and return default arguments as a Namespace object.

        This method initializes a Namespace object with default arguments and assigns
        it to the `self.args_namespace` attribute. Default arguments should be
        defined externally or provided as a dictionary named `default_args`.

        Returns:
            argparse.Namespace: A Namespace object containing default arguments.
        """

        default_args_namespace = Namespace(**default_args)

        self.args_namespace = default_args_namespace

        return default_args
    
    
    # def is_notebook(self):
    #     try:
    #         from IPython import get_python

    #         if "IPKernalApp" not in get_python().config:
    #             raise ImportError("console")
    #         if "VSCODE_PID" in os.environ:
    #             raise ImportError("vscode")
    #             return False
    #     except:
    #         return False
    #     else:
    #         return True

    
    def is_running_locally(self):

        """
        Determine if the code is running locally or in a remote environment.

        This function checks if the `s3_bucket` parameter is provided. If it is empty, it assumes that
        the code is being executed locally. Otherwise, it assumes it's running in a remote environment.

        Returns:
            bool: True if running locally, False if running in a remote environment.

        """

        if self.args['s3_bucket'] == "":
            running_locally = True
        else:
            running_locally = False

        self.args['running_locally'] = running_locally

        return running_locally
    
    
    def determine_data_filepath(self):

        """
         Determine the data filepath based on the execution environment and user input.

        This function calculates the data filepath based on whether the code is running locally
        or in a remote environment and the value of the `data_path` parameter. If the code is
        running remotely, it assumes an empty data path. If running locally, it uses the provided
        `data_path` or raises an exception if `data_path` is not specified.

        Returns:
            data_path (str): The determined data filepath.

        Raises:
            Exception: If `data_path` is not specified when running locally.
        """
        running_locally = self.is_running_locally()

        if not running_locally:
            data_path = ''

        elif running_locally and self.args.get('data_path'):
            data_path = self.args.get('data_path')
        else:
            raise Exception("Need to specify a filepath for data")
        
        self.args['data_path'] = data_path

        return data_path
    