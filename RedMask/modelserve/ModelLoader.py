import mlflow
import redisai
from dotenv import dotenv_values
import tensorflow as tf
tf.enable_control_flow_v2()


class ModelLoader():
    """ Class for conceiving operations with model.
    Only suitable for mlflow tracking server as model storage for now.
    To use - init with server host and port, then upload/download using model  name.
    I used only sklearn flavor because it saves model as BLOB and doesn't require additional info,
    since that info will be required later by redisai anyway.
    """

    def __init__(
            self, tool: str = 'mlflow',
            host: str = 'localhost', port: str = '5000'):  # change to config source
        self.env = dotenv_values('.env')
        self.set_storage(tool, host=self.env['MLFLOW_CONTAINER'], port=self.env['MLFLOW_PORT'])
        self.set_server(host=self.env['REDISAI_CONTAINER'], port=self.env['REDISAI_PORT'])

    def set_storage(
            self, tool:str = 'mlflow', 
            host: str = 'localhost', port: str = '5000'):
        """Connect to Mlflow tracking server """
        if tool == 'mlflow':
            self.tool = tool
            self.host = host
            self.port = port
            self.Client = mlflow.tracking.MlflowClient(f'http://{host}:{port}/')
            mlflow.set_tracking_uri(f'http://{host}:{port}/')
        else:
            raise ValueError("tool: Unknown service")

    def upload_model(
            self, model_name: str, model: bytes,
            params: dict = None, metrics: dict = None, tags: dict = None):
        """Log and register model to Mlflow server
        Experiment name - same as model name
        starts run and save model as BLOB using mlflow.sklearn"""
        self.model_name = model_name
        self.model = model
        mlflow.set_experiment(model_name)
        run = mlflow.start_run()
        # writing params and metrics if there are some
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)

        mlflow.sklearn.log_model(model, model_name)
        self.model_uri = mlflow.get_artifact_uri(model_name)
        reg_model = mlflow.register_model(self.model_uri, model_name)
        self.experiment_id = run.info.run_id
        mlflow.end_run()
        return self.experiment_id

    def download_model(self, model_name: str = '', input_model=None, version: str = 'None'):
        """Download registered model from model storage, latest version if otherwise is not specified.
        Args:
            model_name: name of model, should be the same for different versions of model
            version: None if latest required, otherwise number in str"""
        self.model_name = self.model_name or model_name
        model_uri = f"models:/{self.model_name}/{version}"

        if input_model: print("got imput model")
        print(f"Model name: {model_name}\n")
        # Model must be 'bytes' type
        try:
            input_model = mlflow.sklearn.load_model(model_uri)
            model = bytes(input_model)
        except Exception as e:
            print(repr(e))
        return model

    def set_server(self, host: str, port: str):
        """Connect to RedisAI server"""

        self.redis_client = redisai.Client(host=self.env['REDISAI_CONTAINER'], port=self.env['REDISAI_PORT'])

    def set_model(self,
            key: str, backend: str, device: str, 
            model: bytes = None, inputs: str = None, outputs: str = None):
        """
        Set the model to redisai on provided key.
        """
        self.model = self.model or model

        # Send all the data to the RedisAI server
        model_set = self.redis_client.modelset(key, backend, device, inputs=inputs, outputs=outputs, data=model)
        return model_set

    def updatemodel(self, key: str,  backend: str, device: str,
                    inputs: str = None, outputs: str = None, model=None):
        if model:
            self.model = self.download_model(input_model=model)
        else:
            self.model = self.download_model(key)
        self.set_model(key, backend, device, model=self.model, inputs=inputs, outputs=outputs)
        return 'Model updated'
