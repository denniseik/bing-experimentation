import os
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication

FOLDER_TO_UPLOAD = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../',
        'sample_data'
    )

# load workspace
ws = Workspace.from_config(
    auth=AzureCliAuthentication(),
    path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'config.json'
    )
)

# Retrieve the default datastore in the workspace
datastore = ws.get_default_datastore()

# upload files
datastore.upload(
    src_dir=FOLDER_TO_UPLOAD,
    target_path=None,
    overwrite=True,
    show_progress=True
)
