"""
Model Training Pipeline

Example pipeline with two python script steps and intermediate data.
"""
import os
from azureml.core import Experiment, Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core import RunConfiguration
from azureml.core.authentication import AzureCliAuthentication
from azureml.data.data_reference import DataReference

EXPERIMENT_NAME = "newsgroups_train_pipeline"
COMPUTE_TARGET = "myamlcompute"

# load Azure ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# define datasets
def_blob_store = workspace.get_default_datastore()

input_data = DataReference(
    datastore=def_blob_store,
    data_reference_name="input_data",
    path_on_datastore="20news.pkl"
)

intermediate_data = PipelineData(
    name="model",
    datastore=def_blob_store,
    output_path_on_compute="training"
)

# run configuration
runconfig = RunConfiguration(
    conda_dependencies=CondaDependencies(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../',
            'conda_dependencies.yml'
        )
    )
)

# Definition of pipeline steps
trainStep = PythonScriptStep(
    script_name="train_withdata.py",
    source_directory=os.path.dirname(os.path.realpath(__file__)),
    name="Model Training",
    arguments=[
        '--input_data', str(input_data.as_mount()),
        '--output_data', str(intermediate_data.as_mount())
    ],
    inputs=[
        input_data
    ],
    outputs=[
        intermediate_data
    ],
    compute_target=COMPUTE_TARGET
)

modelConversionStep = PythonScriptStep(
    script_name="model_conversion.py",
    source_directory=os.path.dirname(os.path.realpath(__file__)),
    name="Model Conversion",
    arguments=[
        '--input_data', str(intermediate_data.as_mount())
    ],
    inputs=[
        intermediate_data
    ],
    outputs=[],
    compute_target=COMPUTE_TARGET
)

# Pipeline definition
training_pipeline = Pipeline(
    workspace=workspace,
    steps=[
        trainStep,
        modelConversionStep
    ]
)

# Submit pipeline run
pipeline_run = Experiment(
    workspace,
    EXPERIMENT_NAME
).submit(training_pipeline)

pipeline_run.wait_for_completion()