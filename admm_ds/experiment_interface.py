from abc import ABC, abstractmethod


class ExperimentModel(ABC):
    """
    The purpose of this class is to define the interface our experiment pipeline will expect. This
    should mostly be a thin wrapper around existing data libraries that just standardize on a few
    interfaces for getting the appropriate dataset as well as the run-to-run consistence by ensuring
    the random seed is respected for all experiment samples.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def getModel(self):
        pass

    @abstractmethod
    def getValidationDataset(self):
        pass

    @abstractmethod
    def getTrainingDataset(self):
        pass


class ExperimentTrainer(ABC):
    """
    The purpose of this class is to accept an ExperimentModel and transform it in the various ways
    we need for the purposes of making an experiment.
    """

    def __init__(self, model: ExperimentModel) -> None:
        super().__init__()
        self._model = model

    @abstractmethod
    def heuristic(self):
        """
        This method should inexpensively evaluate the model and return the evaluation heuristic.
        """
        pass

    @abstractmethod
    def train(self, training_args):
        """
        This method should perform training on the model for the supplied training arguments.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        This method should evaluate the model on the validation dataset.
        """
        pass
