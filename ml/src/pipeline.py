import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, help="environment (dev, uat, prd)")
args = parser.parse_args()

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
import os

try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

# Workspace
subscription_id = "59a62e46-b799-4da2-8314-f56ef5acf82b"
resource_group = "rg-azuremltraining"
workspace = "dummy-workspace"
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace,
)



# Data asset creation
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

web_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

credit_data = Data(
    name=f"creditcard_defaults_andrea_{args.env}",
    path=web_path,
    type=AssetTypes.URI_FILE,
    description="Dataset for credit card defaults",
    tags={"source_type": "web", "source": "UCI ML Repo"},
    # version="1.0.0",
)

credit_data = ml_client.data.create_or_update(credit_data)
print(f"Dataset with name {credit_data.name} was registered to workspace, the dataset version is {credit_data.version}")




# Compute cluster
from azure.ai.ml.entities import AmlCompute

cpu_compute_target = "aml-cluster"

cpu_cluster = ml_client.compute.get(cpu_compute_target)
print( f"You already have a cluster named {cpu_compute_target}, we'll reuse it as is.")



# Environment
from azure.ai.ml.entities import Environment
custom_env_name = f"aml-scikit-learn_andrea_{args.env}"

dependencies_dir = "./dependencies"
pipeline_job_env = Environment(
    name=custom_env_name,
    description="Custom environment for Credit Card Defaults pipeline",
    tags={"scikit-learn": "0.24.2"},
    conda_file=os.path.join(dependencies_dir, "conda.yaml"),
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    version="0.1.0",
)
pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

print(f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}")


# Data prep component
from azure.ai.ml import command
from azure.ai.ml import Input, Output

data_prep_src_dir = "./components/data_prep"
data_prep_component = command(
    name=f"data_prep_credit_defaults_andrea_{args.env}",
    display_name="Data preparation for training",
    description="reads a .xl input, split the input to train and test",
    inputs={
        "data": Input(type="uri_file"),
        "test_train_ratio": Input(type="number"),
    },
    outputs=dict(
        train_data=Output(type="uri_folder", mode="rw_mount"),
        test_data=Output(type="uri_folder", mode="rw_mount"),
    ),
    code=data_prep_src_dir,
    command="""python data_prep.py \
            --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} \
            --train_data ${{outputs.train_data}} --test_data ${{outputs.test_data}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

data_prep_component = ml_client.create_or_update(data_prep_component.component)
print(f"Component {data_prep_component.name} with Version {data_prep_component.version} is registered")



# Train Component
from azure.ai.ml import load_component

train_src_dir = "./components/train"
train_component = load_component(source=os.path.join(train_src_dir, "train.yml"))
train_component = ml_client.create_or_update(train_component)
print(f"Component {train_component.name} with Version {train_component.version} is registered")



# Pipeline definition
from azure.ai.ml import dsl, Input, Output

@dsl.pipeline(
    compute=cpu_compute_target,
    description="E2E data_perp-train pipeline",
)
def credit_defaults_pipeline(
    pipeline_job_data_input,
    pipeline_job_test_train_ratio,
    pipeline_job_learning_rate,
    pipeline_job_registered_model_name,
):
    data_prep_job = data_prep_component(
        data=pipeline_job_data_input,
        test_train_ratio=pipeline_job_test_train_ratio,
    )

    train_job = train_component(
        train_data=data_prep_job.outputs.train_data,  # note: using outputs from previous step
        test_data=data_prep_job.outputs.test_data,  # note: using outputs from previous step
        learning_rate=pipeline_job_learning_rate,  # note: using a pipeline input as parameter
        registered_model_name=pipeline_job_registered_model_name,
    )

    return {
        "pipeline_job_train_data": data_prep_job.outputs.train_data,
        "pipeline_job_test_data": data_prep_job.outputs.test_data,
    }

registered_model_name = f"credit_defaults_model_andrea_{args.env}"
pipeline = credit_defaults_pipeline(
    pipeline_job_data_input=Input(type="uri_file", path=credit_data.path),
    pipeline_job_test_train_ratio=0.25,
    pipeline_job_learning_rate=0.05,
    pipeline_job_registered_model_name=registered_model_name,
)


# Submit the pipeline job
import webbrowser

pipeline_job = ml_client.jobs.create_or_update(
    pipeline,
    experiment_name=f"e2e_registered_components_andrea_{args.env}",
)
#webbrowser.open(pipeline_job.studio_url)