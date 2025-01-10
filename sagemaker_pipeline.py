import boto3
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.sklearn.estimator import SKLearn

# Initialize sessions, role, and bucket
region = boto3.Session().region_name
sagemaker_session = sagemaker.Session()
pipeline_session = PipelineSession()
role = "arn:aws:iam::866824485776:role/service-role/AmazonSageMaker-ExecutionRole-20240913T125305"  # Replace with your IAM role ARN
default_bucket = sagemaker_session.default_bucket()

# Define a ScriptProcessor for preprocessing using Scikit-Learn
sklearn_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="0.23-1"),
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    command=["python3"]
)

# Preprocessing Step
processing_step = ProcessingStep(
    name="PreprocessIrisData",
    processor=sklearn_processor,
    code="preprocessing.py",
    inputs=[
        ProcessingInput(
            source=f"s3://{default_bucket}/raw-data/iris.csv",  # Replace with your S3 path to iris.csv
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{default_bucket}/processed-data",
            output_name="processed_data"
        )
    ]
)

# Define an SKLearn estimator for training
sklearn_estimator = SKLearn(
    entry_point="train.py",
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="0.23-1",
    base_job_name="iris-training",
    output_path=f"s3://{default_bucket}/iris-output"
)

# Training Step
training_step = TrainingStep(
    name="TrainIrisModel",
    estimator=sklearn_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                "processed_data"
            ].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)

# Define the pipeline with the preprocessing and training steps
pipeline = Pipeline(
    name="IrisMLOpsPipeline",
    steps=[processing_step, training_step],
    sagemaker_session=sagemaker_session
)

# Create or update the pipeline
pipeline.upsert(role_arn=role)

print("Pipeline for Iris dataset created successfully!")
