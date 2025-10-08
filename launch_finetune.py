# launch.py
import os
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput

role = sagemaker.get_execution_role()

# Your manifests (JSONL with "source-ref" + your label field, e.g., "lang" or "label")
train_manifest_uri = "s3://us-west-2-ehmli/lid-job/manifests/data_train_0930.s3.jsonl"
validation_manifest_uri = "s3://us-west-2-ehmli/lid-job/manifests/data_valid_0930.s3.jsonl"

# Derive the local manifest file names so the training script can find them easily
train_manifest_name = os.path.basename(train_manifest_uri)
eval_manifest_name  = os.path.basename(validation_manifest_uri)

est = PyTorch(
    entry_point="train_lid_hf.py",
    source_dir="src",
    role=role,
    instance_type="ml.g5.2xlarge",
    instance_count=1,
    framework_version="2.3",
    py_version="py311",
    environment={
        # keep HF caches on the big local volume
        "HF_DATASETS_CACHE": "/opt/ml/input/data/hf_cache",
        "TRANSFORMERS_CACHE": "/opt/ml/input/data/hf_cache",
        "TOKENIZERS_PARALLELISM": "false",
    },
    hyperparameters={
        "model_name": "facebook/mms-lid-126",
        "sr": 16000,
        "train_manifest_name": train_manifest_name,
        "eval_manifest_name": eval_manifest_name,
        "label_col": "lang",            # change if your label field is different
        "crop_seconds_train": 2.0,      # training crop length (seconds)
        "crop_seconds_eval": 2.0,       # eval crop length (seconds)
        "num_proc": -1,                 # use all CPUs in map()
        "epochs": 5,
        "batch_size": 16,
        "lr": 5e-5,
        "fp16": True,
        "output_dir": "/opt/ml/model",
    },
    dependencies=["requirements.txt"],
)

inputs = {
    # Each channel is a single MANIFEST file; SageMaker downloads all listed objects to local disk
    "train": TrainingInput(
        s3_data=train_manifest_uri,
        s3_data_type="ManifestFile",
        input_mode="File",
        distribution="FullyReplicated",
    ),
    "validation": TrainingInput(
        s3_data=validation_manifest_uri,
        s3_data_type="ManifestFile",
        input_mode="File",
        distribution="FullyReplicated",
    ),
}

est.fit(inputs)
