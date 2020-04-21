"""
Helper to get run details for debugging purposes
"""
import os
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication


# load workspace
ws = Workspace.from_config(
    auth=AzureCliAuthentication(),
    path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'config.json'
    )
)

print("Datastores in the workspace:")

for ds in ws.datastores:
    print(ds)

print("Default datastore: {}".format(
    ws.get_default_datastore()
)