import sys
import os
import logging

import numpy as np

from admm_ds.compression_configurations import (
    ARGUMENTS_KEY,
    IMPORT_TYPE_KEY,
    HFTransformerADMMConfig,
    TransformerComponent
)
from admm_ds.nxm_admm import NxMQuantizedProjection
from admm_ds.nxm_utils import add_nxm_config
from admm_ds.quantization_utils import add_quantization_config

from datasets import load_dataset, load_metric

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from run_glue_admm import DataTrainingArguments
from admm_transformer_utils import ModelArguments, ADMMTrainer, task_to_keys


logger = logging.getLogger(__name__)


def main() -> None:

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize HF logging system
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if data_args.task_name is None:
        raise Exception("Only GLUE tasks currently supported.")

    datasets = load_dataset("glue", data_args.task_name)

    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    padding = "max_length"

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    dataset_config_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    label_to_id = None
    if (
        dataset_config_model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in dataset_config_model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[label] for label in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]

    metric = load_metric("glue", data_args.task_name)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    configs_to_explore = {}

    for layer in range(12):
        base_config = {}
        entry_style = {
            IMPORT_TYPE_KEY: NxMQuantizedProjection.ProjectionName,
            ARGUMENTS_KEY: add_nxm_config(add_quantization_config({}))
        }
        for gen_layer in range(HFTransformerADMMConfig.numLayers()):
            if layer == gen_layer:
                continue

            key, value, query, atten_output, inter, out = HFTransformerADMMConfig.getAllLayerComponents(gen_layer)
            base_config[key] = entry_style
            base_config[value] = entry_style
            base_config[query] = entry_style
            base_config[atten_output] = entry_style
            base_config[inter] = entry_style
            base_config[out] = entry_style

        ffn_config = dict(base_config)
        TC = TransformerComponent
        ffn_config[HFTransformerADMMConfig.getTransformerModule(layer, TC.KEY)] = entry_style
        ffn_config[HFTransformerADMMConfig.getTransformerModule(layer, TC.QUERY)] = entry_style
        ffn_config[HFTransformerADMMConfig.getTransformerModule(layer, TC.VALUE)] = entry_style
        ffn_config[HFTransformerADMMConfig.getTransformerModule(layer, TC.ATTEN_OUTPUT)] = entry_style

        atten_config = dict(base_config)
        atten_config[HFTransformerADMMConfig.getTransformerModule(layer, TC.INTERMEDIATE)] = entry_style
        atten_config[HFTransformerADMMConfig.getTransformerModule(layer, TC.OUTPUT)] = entry_style

        configs_to_explore["layer_{}_atten".format(layer)] = atten_config
        configs_to_explore["layer_{}_ffn".format(layer)] = ffn_config

    base_path = training_args.output_dir

    print(len(configs_to_explore))

    for config_name, admm_config in configs_to_explore.items():
        logger.info("Running {}".format(config_name))

        set_seed(training_args.seed)
        training_args.output_dir = os.path.join(base_path, config_name)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        trainer = ADMMTrainer(
            model=model,
            admm_args=admm_config,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

        trainer.hard_prune(training_args.output_dir)

        # Evaluation
        eval_results = {}
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Eval results {task} *****")
                    for key, value in sorted(eval_result.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

            eval_results.update(eval_result)


if __name__ == "__main__":
    main()
