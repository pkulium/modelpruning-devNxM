from admm_ds.hf.hf_experiment_model import DataTrainingArguments, HFGlueExperiment, ModelArguments
from admm_ds.hf.hf_training_arguments import admmDSDefaultTrainingArguments

PATH_TO_MODEL = "/scratch/admm_ds/v2/base_checkpoints/mrpc"

data_args = DataTrainingArguments(task_name="mrpc")
training_args = admmDSDefaultTrainingArguments("./test_logs/", 5e-5, 10)
model_args = ModelArguments(PATH_TO_MODEL)

experiment = HFGlueExperiment(data_args, model_args, training_args)

model = experiment.getModel()
train_dataset = experiment.getTrainingDataset()
valid_dataset = experiment.getValidationDataset()

model2 = experiment.getModel()
train_dataset2 = experiment.getTrainingDataset()
valid_dataset2 = experiment.getValidationDataset()
