import json
import logging

from typing import Dict, List, Union

import torch

from .admm_disabled_projection import ADMMDisabledProjection
from .admm_disabled_types import ADMM_DISABLED_TYPES
from .compression_configurations import ARGUMENTS_KEY, IMPORT_TYPE_KEY, TransformerADMMConfig


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ADMMDisabledOptimizer:

    _instances: List[ADMMDisabledProjection]
    _debug_enabled: bool
    _model: torch.nn.Module
    _configuration: Dict

    def __init__(
        self,
        model: torch.nn.Module,
        configuration: Union[str, Dict[str, Dict], TransformerADMMConfig],
        rho=0.001
    ) -> None:
        super(ADMMDisabledOptimizer, self).__init__()

        logger.info("Initializing regular compression approach.")

        self._debug_enabled = False
        self._instances = []
        self._model = model

        if isinstance(configuration, str):
            with open(configuration, "r") as config_file:
                self._configuration = json.load(config_file)
        elif isinstance(configuration, Dict):
            self._configuration = configuration
        elif isinstance(configuration, TransformerADMMConfig):
            self._configuration = configuration.configuration
        else:
            raise Exception("configuration should be name of json config file or configuration Dict")

        for name in self._configuration.keys():
            config: Dict = self._configuration[name]
            projection_class_name = config[IMPORT_TYPE_KEY]
            projection_class = ADMM_DISABLED_TYPES[projection_class_name]
            self._instances.append(projection_class(model, name, config[ARGUMENTS_KEY]))

    def iteration(self) -> None:
        pass

    def regularizer(self) -> torch.Tensor:
        return 0

    def prune_model(self) -> None:
        logger.info("Model pruning enforced by model.")

    def restore_model(self) -> None:
        logger.info("Model state is not outside of training. Restore will do nothing.")

    def get_parameters_for_training(self) -> List[str]:
        """
        Used to get the necessary parameters for training the model such that the gradient optimizer
        will only train the returned parameters.
        """
        params = []
        for instance in self._instances:
            params.extend(instance.get_parameters_for_training())
        return params

    @property
    def configuration(self) -> Dict[str, Dict]:
        return self._configuration

    @property
    def is_hard_pruned(self) -> bool:
        return True
