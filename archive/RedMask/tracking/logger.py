#REDUNDANT SCRIPT

import os
import warnings

#import mlflow
#from mlflow import log_metric, log_param, set_experiment, start_run
#from mlflow.tensorflow import log_model

from RedMask.tracking.remote_server import RemoteTracking
from RedMask.training.training_funcs import train_model

warnings.filterwarnings("ignore")

#mlflow.tensorflow.autolog()

class FlowTraining:

    def __init__(self, tracking_uri, device):
        self.tracking_uri = tracking_uri
        self.remote_server = RemoteTracking(tracking_uri=tracking_uri)
        self.local_experiment_dir = './mlruns'
        self.local_experiment_id = '0'
        self.device = device

    def log_tags_and_params(self, remote_run_id):
        run_id = self.get_local_run_id()
        mlflow.set_tracking_uri(self.local_experiment_dir)
        run = mlflow.get_run(run_id=remote_run_id)
        params = run.data.params
        tags = run.data.tags
        self.remote_server.set_tags(remote_run_id, tags)
        self.remote_server.log_params(remote_run_id, params)

    def get_local_run_id(self):
        files = os.listdir(os.path.join(self.local_experiment_dir, self.local_experiment_id))
        for file in files:
            if not file.endswith('.yaml'):
                return file

    def run(self, dataset, experiment_name, art_loc):
        """
        Runs experiment.

        Parameters
        ----------
        dataset : str
            'small' of 'full'.
        experiment_name : str
            Experiment name.
        art_loc : str
            Artifact location. If empty, it will be a local directory.

        Returns
        -------
        None.

        """
        # getting the id of the experiment, creating an experiment in its absence
        remote_experiment_id, remote_experiment = self.remote_server.get_experiment_id(name=experiment_name,
                                                                                       artifact_location=art_loc)
        # creating a "run" and getting its id
        remote_run_id = self.remote_server.get_run_id(remote_experiment_id)

        # indicate that we want to save the results on a remote server
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)

        print("Experiment_id: {}".format(remote_experiment.experiment_id))
        print("Artifact Location: {}".format(remote_experiment.artifact_location))
        print("Tags: {}".format(remote_experiment.tags))
        print("Lifecycle_stage: {}".format(remote_experiment.lifecycle_stage))

        with mlflow.start_run(run_id=remote_run_id, nested=False) as run:
            model, accuracy = train_model(self.device, dataset)
            print(f"run ID is  {run.info.run_id}")

            #Log metric
            mlflow.log_metric('Categorical accuracy', accuracy)
            # Log an artifact (output file)
            mlflow.log_artifact('../output/frozen_model_1.15.0.pb')
            #mlflow.log_artifact('../output/tflite_model_tf_1_15.tflite')
            print("Models saved in run %s" % mlflow.active_run().info.run_uuid)

        # Register model version
        model_uri = "runs:/{}/redmask_model".format(run.info.run_id)
        mv = mlflow.register_model(model_uri, "redmask_model")
