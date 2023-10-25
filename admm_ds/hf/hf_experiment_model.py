import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from datasets import load_dataset, load_metric

from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    PretrainedConfig,
    default_data_collator,
    TrainingArguments,
    set_seed
)

from admm_ds.experiment_interface import ExperimentModel


TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(TASK_TO_KEYS.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in TASK_TO_KEYS.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(TASK_TO_KEYS.keys()))
        else:
            raise ValueError("Need a GLUE task.")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )


class HFGlueExperiment(ExperimentModel):

    def __init__(self, data_args: DataTrainingArguments, model_args: ModelArguments, training_args: TrainingArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        datasets = load_dataset("glue", data_args.task_name)
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1

        padding = "max_length"
        sentence1_key, sentence2_key = TASK_TO_KEYS[data_args.task_name]

        self.config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=self.config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision
        )

        label_to_id = None
        if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and not is_regression
        ):
            label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
            if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
                label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
            else:
                logger.warn(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                    "\nIgnoring the model labels as a result.",
                )

        max_seq_length = min(data_args.max_seq_length, self.tokenizer.model_max_length)

        def preprocess_function(examples):
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = self.tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

            if label_to_id is not None and "label" in examples:
                result["label"] = [label_to_id[label] for label in examples["label"]]
            return result

        self.datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=True)

        self.train_dataset = self.datasets["train"]
        self.eval_dataset = self.datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        self.test_dataset = self.datasets["test_matched" if data_args.task_name == "mnli" else "test"]

        metric = load_metric("glue", data_args.task_name)

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        self.compute_metrics = compute_metrics

        self.data_collator = default_data_collator

    def getModel(self):
        set_seed(self.training_args.seed)
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_args.model_name_or_path,
            config=self.config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision
        )

    def getTrainingDataset(self):
        return self.train_dataset

    def getValidationDataset(self):
        return self.eval_dataset
