"""
Helper to get run details for debugging
"""
import os
from azureml.core import Workspace, Experiment, Run
from azureml.core.authentication import AzureCliAuthentication

EXPERIMENT = "test"
RUNID = "123"

# load workspace
ws = Workspace.from_config(
    auth=AzureCliAuthentication(),
    path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'config.json'
    )
)
experiment = Experiment(
    workspace=ws,
    name=EXPERIMENT
)

run = Run(
    experiment,
    run_id=RUNID
)

print("Run details: {}".format(
    run.get_details()
)

print("Run metrics: {}".format(
    run.get_metrics()
)