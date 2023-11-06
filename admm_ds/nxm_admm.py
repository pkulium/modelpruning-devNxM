import logging
from typing import Dict

import torch

from admm_ds.layers import FakeQuantizedFullyConnectedLayer, FakeQuantizedWeightsFullyConnectedLayer
from admm_ds.quantization_utils import (
    add_quantization_config,
    fixedOffsetQuantizationSearch,
    load_quantization_config,
    quantizeSymmetricSimple
)
from .admm_projection import ADMMFCLProjection, ADMMQuantizedFCLProjection, ADMMFCLProjection_Lora
from .nxm_utils import (
    add_nxm_config,
    maskNxM,
    load_nxm_config
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NxMProjection(ADMMFCLProjection):
    """
    Projection handler for N:M semi-structured pruning on fully-connected layers.
    """
    ProjectionName = "NxM"

    def __init__(self,
                 model: torch.nn.Module,
                 targeted_module: str,
                 compression_args: Dict) -> None:
        super(NxMProjection, self).__init__(model, targeted_module, compression_args)

        self._n, self._m = load_nxm_config(compression_args, logger)

    def project(self) -> None:
        with torch.no_grad():
            values = self._module.weight.data + self._u
            scores = values.abs()

            mask = maskNxM(scores, self._n, self._m)
            self._z = mask * values

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

class NxMProjection_Lora(ADMMFCLProjection_Lora):
    """
    Projection handler for N:M semi-structured pruning on fully-connected layers.
    """
    ProjectionName = "NxM_Lora"

    def __init__(self,
                 model: torch.nn.Module,
                 targeted_module: str,
                 compression_args: Dict) -> None:
        super(NxMProjection_Lora, self).__init__(model, targeted_module, compression_args)

        self._n, self._m = load_nxm_config(compression_args, logger)

    def project(self) -> None:
        with torch.no_grad():
            values = self._module.weight.data + self._module.lora_B.data @ self._module.lora_A.data + self._u
            scores = values.abs()

            mask = maskNxM(scores, self._n, self._m)
            self._z = mask * values

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


class NxMQuantizedProjection(ADMMQuantizedFCLProjection):
    """
    Projection handler for combined N:M semi-structured sparsity with symmetric
    quantization on fully-connected layers.
    """
    ProjectionName = "NxM_QUANT"
    _module: FakeQuantizedFullyConnectedLayer

    def __init__(self,
                 model: torch.nn.Module,
                 targeted_module: str,
                 compression_args: Dict) -> None:
        super(NxMQuantizedProjection, self).__init__(model, targeted_module, compression_args)

        self._n, self._m = load_nxm_config(compression_args, logger)
        self._num_bits, self._norm, self._chunk_size, self._num_groups = load_quantization_config(compression_args, logger)
        self._module.weights_bits = self._num_bits

    def project(self) -> None:
        with torch.no_grad():
            if self._u.device != self._module.weight.data.device:
                print(self._name)

            values = self._module.weight.data + self._u
            scores = values.abs()

            mask = maskNxM(scores, self._n, self._m)
            values = mask * values

            _, self._z = fixedOffsetQuantizationSearch(
                values,
                self._num_bits,
                0,
                self._chunk_size,
                self._norm
            )

    @classmethod
    def createConfig(cls, n: int, m: int, num_bits: int, chunk_size: int) -> Dict:
        from admm_ds.compression_configurations import TransformerADMMConfig

        return TransformerADMMConfig.buildEntry(
            cls,
            add_quantization_config(add_nxm_config({}, n=n, m=m), num_bits=num_bits, chunk_size=chunk_size)
        )

    @staticmethod
    def compression_ratio(compression_args: Dict) -> float:
        n, m = load_nxm_config(compression_args, logger)
        num_bits, _, _, _ = load_quantization_config(compression_args, logger)
        return (float(m) * float(num_bits + 2)) / (float(n) * float(32))


class NxMModifiedQuantizedProjection(ADMMQuantizedFCLProjection):
    """
    Projection handler for combined N:M semi-structured sparsity with symmetric
    quantization on fully-connected layers.

    This differs from NxMQuantizedProjection by using the non-search based quantization
    approach for solving the projection.
    """
    ProjectionName = "NxM_MOD_QUANT"

    _module: FakeQuantizedFullyConnectedLayer

    def __init__(
        self,
        model: torch.nn.Module,
        targeted_module: str,
        compression_args: Dict
    ) -> None:
        super(NxMModifiedQuantizedProjection, self).__init__(model, targeted_module, compression_args)

        self._n, self._m = load_nxm_config(compression_args, logger)
        self._num_bits, self._norm, self._chunk_size, self._num_groups = load_quantization_config(compression_args, logger)
        self._module.weights_bits = self._num_bits

    def project(self) -> None:
        with torch.no_grad():
            if self._u.device != self._module.weight.data.device:
                print(self._name)

            values = self._module.weight.data + self._u
            scores = values.abs()

            mask = maskNxM(scores, self._n, self._m)
            values = mask * values

            self._z = quantizeSymmetricSimple(
                values,
                self._num_bits,
                self._chunk_size,
            )

    @classmethod
    def createConfig(cls, n: int, m: int, num_bits: int, chunk_size: int) -> Dict:
        from admm_ds.compression_configurations import TransformerADMMConfig

        return TransformerADMMConfig.buildEntry(
            cls,
            add_quantization_config(add_nxm_config({}, n=n, m=m), num_bits=num_bits, chunk_size=chunk_size)
        )

    @staticmethod
    def compression_ratio(compression_args: Dict) -> float:
        n, m = load_nxm_config(compression_args, logger)
        num_bits, _, _, _ = load_quantization_config(compression_args, logger)
        return (float(m) * float(num_bits + 2)) / (float(n) * float(32))


class NxMSTEQuantizedProjection(ADMMFCLProjection):

    ProjectionName = "NxM_ADMM_STE"

    _module = FakeQuantizedWeightsFullyConnectedLayer

    def __init__(self, model: torch.nn.Module, targeted_module: str, compression_args: Dict) -> None:
        super(NxMSTEQuantizedProjection, self).__init__(model, targeted_module, compression_args)

        self._n, self._m = load_nxm_config(compression_args, logger)
        num_bits, _, chunk_size, _ = load_quantization_config(compression_args, logger)

        # We assume that the entire module is not encapsulated by our single module. This is a valid
        # assumption for our uses cases, but isn't fully general
        parent_module_name = self._name[:self._name.rfind(".")]
        attribute_name = self._name[self._name.rfind(".")+1:]

        parent_module = model.get_submodule(parent_module_name)
        new_module = FakeQuantizedWeightsFullyConnectedLayer(
            self._module.weight,
            bias=self._module.bias,
            weight_bits=num_bits,
            chunk_size=chunk_size
        )

        setattr(parent_module, attribute_name, new_module)

        self._module = model.get_submodule(self._name)

    def project(self) -> None:
        with torch.no_grad():
            values = self._module.weight.data + self._u
            scores = values.abs()

            mask = maskNxM(scores, self._n, self._m)
            self._z = mask * values

    @classmethod
    def createConfig(cls, n: int, m: int, num_bits: int, chunk_size: int) -> Dict:
        from admm_ds.compression_configurations import TransformerADMMConfig

        return TransformerADMMConfig.buildEntry(
            cls,
            add_quantization_config(add_nxm_config({}, n=n, m=m), num_bits=num_bits, chunk_size=chunk_size)
        )

    @staticmethod
    def compression_ratio(compression_args: Dict) -> float:
        n, m = load_nxm_config(compression_args, logger)
        num_bits, _, _, _ = load_quantization_config(compression_args, logger)
        return (float(m) * float(num_bits + 2)) / (float(n) * float(32))
