"""
Inheritance of ADMM projection that only implements stubs to remove their functionality
such that the only key parts run are that of the module replacement.
"""

import logging
from abc import abstractmethod
from typing import List, Optional, Dict

import torch

from admm_ds.admm_projection import ADMMProjection
from admm_ds.layers import (
    FakeQuantizedWeightsFullyConnectedLayer,
    MaskedFakeQuantizedWeightsFullyConnectedLayer,
    MaskedFullyConnectedLayer
)
from admm_ds.nxm_utils import add_nxm_config, load_nxm_config
from admm_ds.quantization_utils import add_quantization_config, load_quantization_config
from admm_ds.admm_disabled_error import ADMMDisabledError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ADMMDisabledProjection(ADMMProjection):
    """
    Base class for compression without ADMM. This class maintains the same interface as an ADMMProjection
    to enable some shared behavior for fetching the parent module, but otherwise implements stubs that
    should never be called by an ADMMDisabledOptimizer.

    Sub-classing this for implementations consists of configuring the modified injected module correctly
    for the desired compression.
    """

    _module: torch.nn.Module
    _name: str
    _u: Optional[torch.tensor]
    _z: Optional[torch.tensor]
    _cached_weight: Optional[torch.tensor]

    @abstractmethod
    def __init__(self, model: torch.nn.Module, targeted_module: str, compression_args: Dict):
        super(ADMMDisabledProjection, self).__init__(model, targeted_module, compression_args)

    def project(self) -> None:
        raise ADMMDisabledError("project")

    def loss(self) -> torch.Tensor:
        raise ADMMDisabledError("loss")

    def update_u(self) -> None:
        raise ADMMDisabledError("update_u")

    def prune_module(self) -> None:
        raise ADMMDisabledError("prune_model")

    def restore_module(self) -> None:
        raise ADMMDisabledError("restore_module")

    def match_parameter_device(self) -> None:
        pass

    def get_parameters_for_training(self) -> List[str]:
        return super().get_parameters_for_training()


class ADMMDisabledNxMCompressor(ADMMDisabledProjection):

    ProjectionName = "NxM_ASP"

    _module: MaskedFullyConnectedLayer

    def __init__(self, model: torch.nn.Module, targeted_module: str, compression_args: Dict) -> None:
        super(ADMMDisabledNxMCompressor, self).__init__(model, targeted_module, compression_args)

        sparse_n, sparse_m = load_nxm_config(compression_args, logger)

        # We assume that the entire module is not encapsulated by our single module. This is a valid
        # assumption for our uses cases, but isn't fully general
        parent_module_name = self._name[:self._name.rfind(".")]
        attribute_name = self._name[self._name.rfind(".")+1:]

        # Replace the Linear layer with a masked FCL quantized layer
        parent_module = model.get_submodule(parent_module_name)
        new_module = MaskedFullyConnectedLayer(
            self._module.weight,
            bias=self._module.bias,
            sparse_n=sparse_n,
            sparse_m=sparse_m
        )
        setattr(parent_module, attribute_name, new_module)

        # Update our reference to get the new submodule in the model.
        self._module = model.get_submodule(self._name)

    @classmethod
    def createConfig(cls, n: int, m: int) -> Dict:
        from admm_ds.compression_configurations import TransformerADMMConfig

        return TransformerADMMConfig.buildEntry(
            cls,
            add_nxm_config({}, n=n, m=m)
        )

    @staticmethod
    def compression_ratio(compression_args: Dict) -> float:
        n, m = load_nxm_config(compression_args, logger)
        return (float(m) * float(16 + 2)) / (float(n) * float(32))


class ADMMDisabledMaskedQuantizedCompressor(ADMMDisabledProjection):

    ProjectionName = "NxM_QAT"

    _module: MaskedFakeQuantizedWeightsFullyConnectedLayer

    def __init__(self, model: torch.nn.Module, targeted_module: str, compression_args: Dict) -> None:
        super(ADMMDisabledMaskedQuantizedCompressor, self).__init__(model, targeted_module, compression_args)
        sparse_n, sparse_m = load_nxm_config(compression_args, logger)
        num_bits, _, chunk_size, _ = load_quantization_config(compression_args, logger)

        # We assume that the entire module is not encapsulated by our single module. This is a valid
        # assumption for our uses cases, but isn't fully general
        parent_module_name = self._name[:self._name.rfind(".")]
        attribute_name = self._name[self._name.rfind(".")+1:]

        parent_module = model.get_submodule(parent_module_name)
        new_module = MaskedFakeQuantizedWeightsFullyConnectedLayer(
            self._module.weight,
            bias=self._module.bias,
            sparse_n=sparse_n,
            sparse_m=sparse_m,
            weights_bits=num_bits,
            chunk_size=chunk_size
        )

        setattr(parent_module, attribute_name, new_module)

        # Update our reference to get the new submodule in the model.
        self._module = model.get_submodule(self._name)

    @classmethod
    def createConfig(cls, n: int, m: int, num_bits: int, chunk_size: int) -> Dict:
        from admm_ds.compression_configurations import TransformerADMMConfig

        return TransformerADMMConfig.buildEntry(
            cls,
            add_nxm_config(add_quantization_config({}, num_bits=num_bits, chunk_size=chunk_size), n=n, m=m)
        )

    @staticmethod
    def compression_ratio(compression_args: Dict) -> float:
        n, m = load_nxm_config(compression_args, logger)
        num_bits, _, _, _ = load_quantization_config(compression_args, logger)
        return (float(m) * float(num_bits + 2)) / (float(n) * float(32))


class ADMMDisabledSTECompressor(ADMMDisabledProjection):

    ProjectionName = "STE"

    _module: FakeQuantizedWeightsFullyConnectedLayer

    def __init__(self, model: torch.nn.Module, targeted_module: str, compression_args: Dict) -> None:
        super(ADMMDisabledSTECompressor, self).__init__(model, targeted_module, compression_args)

        num_bits, _, chunk_size, _ = load_quantization_config(compression_args, logger)

        # We assume that the entire module is not encapsulated by our single module. This is a valid
        # assumption for our uses cases, but isn't fully general
        parent_module_name = self._name[:self._name.rfind(".")]
        attribute_name = self._name[self._name.rfind(".")+1:]

        parent_module = model.get_submodule(parent_module_name)
        new_module = FakeQuantizedWeightsFullyConnectedLayer(
            self._module.weight,
            self._module.bias,
            weight_bits=num_bits,
            chunk_size=chunk_size
        )

        setattr(parent_module, attribute_name, new_module)

        # Update our reference to get the new submodule in the model.
        self._module = model.get_submodule(self._name)

    @classmethod
    def createConfig(cls, num_bits: int, chunk_size: int) -> Dict:
        from admm_ds.compression_configurations import TransformerADMMConfig

        return TransformerADMMConfig.buildEntry(
            cls,
            add_quantization_config({}, num_bits=num_bits, chunk_size=chunk_size)
        )

    @staticmethod
    def compression_ratio(compression_args: Dict) -> float:
        num_bits, _, _, _ = load_quantization_config(compression_args, logger)
        return float(num_bits) / float(32)
