import logging
import os
import shutil
import sys
from dataclasses import dataclass, field

from transformers import HfArgumentParser

from admm_ds.hf.hf_experiment_model import DataTrainingArguments, HFGlueExperiment, ModelArguments
from admm_ds.hf.hf_training_arguments import admmDSDefaultEvalArguments
from admm_ds.hf.hf_trainer import ADMMArguments, ADMMTrainer
from admm_ds.compression_configurations import SearchExperimentConfig

logger = logging.getLogger(__name__)

TASK_TO_METRIC = {
    "rte": "eval_accuracy",
    "mrpc": "eval_f1",
    "stsb": "eval_pearson",
    "cola": "eval_matthews_correlation",
    "sst2": "eval_accuracy",
    "qnli": "eval_accuracy",
    "mnli": "eval_accuracy",
    "qqp": "eval_accuracy"
}


class TopKContainer:

    def __init__(self, k) -> None:
        self._container = []
        self._k = k

    def insert(self, score, config) -> bool:
        if self.full:
            if score < self._container[0][0]:
                return False
            else:
                self._container.pop(0)
                for index, value in enumerate(self._container):
                    if value[0] < score:
                        continue
                    self._container.insert(index, (score, config))
                    return True
                self._container.insert(len(self._container), (score, config))
                return True
        else:
            for index, value in enumerate(self._container):
                if value[0] < score:
                    continue
                self._container.insert(index, (score, config))
                return True
            self._container.insert(len(self._container), (score, config))
            return True

    @property
    def full(self):
        return len(self._container) == self._k

    def __iter__(self):
        self._iter = iter(self._container)
        return self._iter

    def __next__(self):
        return next(self._iter)

    def __repr__(self):
        return "Size {}: {}".format(self._k, self._container)


@dataclass
class ExperimentArguments:
    """
    Arguments pertaining to the experiment search space.
    """

    experiment: str = field(
        metadata={"help": "Path to json file containing experiment parameters."}
    )

    experiment_dir: str = field(
        metadata={"help": "Path the experiment should use as its root for saving results."}
    )

    top_k: int = field(
        default=5,
        metadata={"help": "How many configurations should be retained for full compression."}
    )


def main():

    parser = HfArgumentParser((ModelArguments, ExperimentArguments, DataTrainingArguments))
    model_args, experiment_args, data_args = parser.parse_args_into_dataclasses()
    training_args = admmDSDefaultEvalArguments(experiment_args.experiment_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # We will get datasets and models from the experiment manager
    experiment_manager = HFGlueExperiment(data_args, model_args, training_args)

    # Enumerates our search space
    experiment_config = SearchExperimentConfig(experiment_args.experiment)
    hidden_size, intermediate_size = experiment_config.modelDimensions()

    kept_configs = TopKContainer(experiment_args.top_k)

    for config_id, config in experiment_config:
        if config.compressionRatio(hidden_size, intermediate_size) <= experiment_config.compressionTarget():
            model = experiment_manager.getModel()
            training_dataset = experiment_manager.getTrainingDataset()
            validation_dataset = experiment_manager.getValidationDataset()
            instance_dir = os.path.join(experiment_args.experiment_dir, str(config_id))
            training_args = admmDSDefaultEvalArguments(instance_dir)

            admm_args = ADMMArguments(
                do_compression=True,
                admm_config=config
            )

            trainer = ADMMTrainer(
                admm_args,
                model=model,
                args=training_args,
                train_dataset=training_dataset,
                eval_dataset=validation_dataset,
                compute_metrics=experiment_manager.compute_metrics,
                tokenizer=experiment_manager.tokenizer,
                data_collator=experiment_manager.data_collator
            )

            trainer.hard_prune()

            eval_results = {}
            logger.info("*** Evaluate ***")

            # Loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = [data_args.task_name]
            eval_datasets = [validation_dataset]

            for eval_dataset, _ in zip(eval_datasets, tasks):
                eval_result = trainer.evaluate(eval_dataset=eval_dataset)
                eval_results.update(eval_result)

            kept_configs.insert(
                eval_results[TASK_TO_METRIC[data_args.task_name]],
                (config_id, config)
            )

            # The trainer automatically creates an empty directory even though we don't end up saving
            # anything there. We'll delete that here.
            shutil.rmtree(instance_dir)

    for (score, (config_id, config)) in kept_configs:

        # Save the config for use by actual compression framework.
        save_loc = os.path.join(experiment_args.experiment_dir, "{}.json".format(config_id))
        config.saveConfig(save_loc)
        logger.info("Saving config {} with score {}".format(config_id, score))


if __name__ == "__main__":
    main()
