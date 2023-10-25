import transformers
from transformers import TrainingArguments


def admmDSDefaultTrainingArguments(
    path_to_save: str,
    learning_rate: float,
    num_train_epochs: int
):
    """
    Convenience function that passes in the constant values we will use for all
    of our ADMM-based programmatic experiments.
    """
    return TrainingArguments(
        path_to_save,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=128,
        logging_steps=2000,
        evaluation_strategy=transformers.trainer_utils.IntervalStrategy.EPOCH,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        save_steps=10000
    )


def admmDSDefaultEvalArguments(
    path_to_save: str
):
    """
    Convenience functin that passes in the constant values for when we are only doing
    evaluation. This is used when evaluating each candidate configuration.
    """
    return TrainingArguments(
        path_to_save,
        do_eval=True,
        per_device_eval_batch_size=1024
    )
