"""
Training submitter

Facilitates (remote) training execution through the Azure ML service.
"""
import os
import numpy as np
from azureml.core import Workspace, Experiment
from azureml.train.estimator import Estimator
from azureml.core.authentication import AzureCliAuthentication

EXPERIMENT_NAME = "diabetes_train_multiple"
COMPUTE_TARGET = "local"  # local / myamlcompute

# load Azure ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# Define the Run Configuration
# Azure ML can manage environments for you across compute targets. Since this may take a long time,
# we will set the flags use_docker=False and user_managed=True, so that Azure ML will use our local environment.
est = Estimator(
    entry_script='train_multiple.py',
    source_directory=os.path.dirname(os.path.realpath(__file__)),
    compute_target=COMPUTE_TARGET,
    script_params={
        '--alphas': np.arange(0.0, 1.0, 0.05)
    },
    use_docker=False,
    user_managed=True,
    # conda_dependencies_file=os.path.join(
    #     os.path.dirname(os.path.realpath(__file__)),
    #     '../',
    #     'conda_dependencies.yml'
    # )
)

# Define the ML experiment
experiment = Experiment(workspace, EXPERIMENT_NAME)

# Submit experiment run, if compute is idle, this may take some time
run = experiment.submit(est)

# wait for run completion of the run, while showing the logs
run.wait_for_completion(show_output=True)
