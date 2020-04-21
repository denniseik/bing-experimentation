"""
Training submitter

Facilitates (remote) training execution through the Azure ML service.
"""
import os
from azureml.core import Workspace, Experiment
from azureml.train.estimator import Estimator
from azureml.core.authentication import AzureCliAuthentication
from azureml.data.data_reference import DataReference

EXPERIMENT_NAME = "train_with_data_example"
COMPUTE_TARGET = "myamlcompute" # local / myamlcompute

# load Azure ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# Define the Datasets that we would like to pass with our run
def_blob_store = workspace.get_default_datastore()
input_data = DataReference(
    datastore=def_blob_store,
    data_reference_name="input_data",
    path_on_datastore="20news.pkl"
)

# Define the Run Configuration
# Azure ML can manage environments for you across compute targets. Since this may take a long time,
# we will set the flags use_docker=False and user_managed=True, so that Azure ML will use our local environment.
est = Estimator(
    entry_script='train_withdata.py',
    source_directory=os.path.dirname(os.path.realpath(__file__)),
    compute_target=COMPUTE_TARGET,
    script_params={  
        '--input_data': input_data.as_mount()
    },
    inputs=[
        input_data
    ]
)

# Define the ML experiment
experiment = Experiment(workspace, EXPERIMENT_NAME)

# Submit experiment run, if compute is idle, this may take some time
run = experiment.submit(est)

# wait for run completion of the run, while showing the logs
run.wait_for_completion(show_output=True)
