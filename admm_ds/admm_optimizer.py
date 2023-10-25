import json
import logging
from typing import Dict, List, Union

import torch

from admm_ds.admm_disabled_projection import ADMMDisabledProjection

from .admm_projection import ADMMProjection
from .admm_types import ADMM_TYPES
from .compression_configurations import ARGUMENTS_KEY, IMPORT_TYPE_KEY, TransformerADMMConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ADMMOptimizer:

    _instances: List[ADMMProjection]
    _rho: float
    _debug_enabled: bool
    _model: torch.nn.Module
    _configuration: Dict
    _is_hard_pruned: bool

    def __init__(self,
                 model: torch.nn.Module,
                 configuration: Union[str, Dict[str, Dict], TransformerADMMConfig],
                 rho=0.001) -> None:

        super(ADMMOptimizer, self).__init__()

        logger.info("Initializing optimizer")

        self._rho = rho
        self._debug_enabled = False
        self._instances = []
        self._model = model
        self._is_hard_pruned = False

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
            projection_class = ADMM_TYPES[projection_class_name]
            self._instances.append(projection_class(model, name, config[ARGUMENTS_KEY]))

        self.iteration()

    def iteration(self) -> None:
        """
        Tell all ADMMProjection instances modify their ADMM variables according to their configuration.
        """
        for instance in self._instances:
            instance.match_parameter_device()

            if isinstance(instance, ADMMDisabledProjection):
                continue

            instance.project()
            instance.update_u()

    def regularizer(self) -> torch.Tensor:
        """
        Collect ADMM loss from each ADMMProjection instance and backpropagate the loss. The loss
        is also returned if it is necessary to be used.
        """
        loss = 0
        for instance in self._instances:
            instance.match_parameter_device()

            if isinstance(instance, ADMMDisabledProjection):
                continue

            loss += 0.5 * self._rho * instance.loss()

        if isinstance(loss, torch.Tensor):
            loss.backward()
        return loss

    def prune_model(self) -> None:
        """
        Clamp each compressed module to its constraints. The state of ADMM optimizers and the
        most recent parameter values will be persisted.
        """
        logger.info("Pruning model")

        for instance in self._instances:
            if isinstance(instance, ADMMDisabledProjection):
                continue

            instance.prune_module()

        self._is_hard_pruned = True

    def restore_model(self) -> None:
        """
        Revert each module's state to its last cached state, either the original model state or the
        most recent state before a prune_model call. Note that doing training and then rolling back
        with restore model is likely to corrupt the ADMM state.
        """
        for instance in self._instances:
            if isinstance(instance, ADMMDisabledProjection):
                continue

            instance.restore_module()

        # The conditional here is not strictly necessary, but I want to leave it in as a reminder
        # to revisit the logic about this.
        if self._is_hard_pruned:
            self._is_hard_pruned = False

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
        return self._is_hard_pruned
