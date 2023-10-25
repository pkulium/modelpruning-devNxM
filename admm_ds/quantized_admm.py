import logging
from typing import Dict

import torch

from .admm_projection import ADMMQuantizedFCLProjection, ADMMMaskedQuantizedFCLProjection
from .layers import FakeQuantizedFullyConnectedLayer, MaskedFakeQuantizedFullyConnectedLayer

from .quantization_utils import (
    add_quantization_config,
    fixedOffsetQuantizationSearch,
    load_quantization_config
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SymmetricQuantizedProjection(ADMMQuantizedFCLProjection):

    ProjectionName = "QUANT"

    _module: FakeQuantizedFullyConnectedLayer
    _num_bits: int
    _norm: int
    _chunk_size: int
    _num_groups: int

    def __init__(self, model: torch.nn.Module, targeted_module: str, compression_args: Dict) -> None:
        super(SymmetricQuantizedProjection, self).__init__(model, targeted_module, compression_args)

        self._num_bits, self._norm, self._chunk_size, self._num_groups = load_quantization_config(compression_args, logger)
        self._module.weights_bits = self._num_bits

    def project(self) -> None:
        with torch.no_grad():
            _, self._z = fixedOffsetQuantizationSearch(
                self._module.weight.data + self._u,
                self._num_bits,
                0,
                self._chunk_size,
                self._norm
            )

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


class SymmetricMaskedQuantizedProjection(ADMMMaskedQuantizedFCLProjection):
    """
    Same behavior as ``SymmetricQuantizedProjection`` but will mask with NxM semi-structured sparsity
    under the assumption that it is the second phase of two step compression.
    """

    ProjectionName = "MASKED_QUANT"

    _module: MaskedFakeQuantizedFullyConnectedLayer
    _num_bits: int
    _norm: int
    _chunk_size: int
    _num_groups: int

    def __init__(self, model: torch.nn.Module, targeted_module: str, compression_args: Dict) -> None:
        super(SymmetricMaskedQuantizedProjection, self).__init__(model, targeted_module, compression_args)

        self._num_bits, self._norm, self._chunk_size, self._num_groups = load_quantization_config(compression_args, logger)
        self._module.weights_bits = self._num_bits

    def project(self) -> None:
        with torch.no_grad():
            _, self._z = fixedOffsetQuantizationSearch(
                self._module.weight.data + self._u,
                self._num_bits,
                0,
                self._chunk_size,
                self._norm
            )

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
        return float(num_bits + 2) / (2 * float(32))
