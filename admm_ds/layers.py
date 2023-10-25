from typing import Optional

import torch
import torch.nn.functional as F

from admm_ds.nxm_utils import maskNxM
from admm_ds.quantization_utils import quantizeSymmetricSimple


class SymQuantizer(torch.autograd.Function):
    """
    Symmetric quantization with STE.
    """

    @staticmethod
    def forward(
        ctx,
        input,
        num_bits,
        chunk_size
    ):
        return quantizeSymmetricSimple(input, num_bits, chunk_size)

    @staticmethod
    def backward(
        ctx,
        grad_output
    ):
        grad_input = grad_output.clone()
        return grad_input, None, None


class FakeQuantizedFullyConnectedLayer(torch.nn.Module):

    weight: torch.Tensor
    bias: Optional[torch.Tensor]
    _out_features: int
    _in_features: int
    _weights_bits: int
    _activation_bits: int

    def __init__(
        self,
        weight: torch.nn.Parameter,
        bias=None,
        activation_bits: int = 8,
        weights_bits: int = 8
    ) -> None:
        super(FakeQuantizedFullyConnectedLayer, self).__init__()

        self.weight = weight
        self.bias = bias
        self._out_features, self._in_features = self.weight.shape

        self._weights_bits = weights_bits
        self._activation_bits = activation_bits

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = SymQuantizer.apply(input, self._activation_bits, 1)

        return F.linear(input, self.weight, bias=self.bias)

    @property
    def weights_bits(self) -> int:
        return self._weights_bits

    @weights_bits.setter
    def weights_bits(self, new_value: int) -> None:
        self._weights_bits = new_value

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, weights_bits={}, activation_bits={}".format(
            self._in_features, self._out_features, self.bias is not None, self._weights_bits, self._activation_bits
        )


class MaskedFullyConnectedLayer(torch.nn.Module):
    """
    This FCL is used when doing non-ADMM sparsification by enforcing a statically determined
    sparse mask on every forward pass. Since the masking occurs outside of the gradients, there
    should be no effects.
    """

    weight: torch.Tensor
    _mask: torch.Tensor
    bias: Optional[torch.Tensor]
    _out_features: int
    _in_features: int

    def __init__(
        self,
        weight: torch.nn.Parameter,
        bias=None,
        sparse_n: int = 4,
        sparse_m: int = 2
    ) -> None:
        super(MaskedFullyConnectedLayer, self).__init__()

        self.weight = weight
        self.bias = bias

        self._out_features, self._in_features = self.weight.shape

        self._mask = maskNxM(self.weight.abs(), sparse_n, sparse_m)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight.data = self.weight.data * self._mask

        return F.linear(input, self.weight, bias=self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self._in_features, self._out_features, self.bias is not None
        )


class MaskedFakeQuantizedFullyConnectedLayer(FakeQuantizedFullyConnectedLayer):
    """
    This implementation of the FakeQuantized FCL is useful for when doing ADMM in sequence
    with NxM quantization of the model. On model instantiation, a mask is calculated that will
    mask the weight matrix whenever the forward mechanism is called. This is a sub-optimal
    strategy if something like gradient accumulation is being used, but the overhead should
    be minimal.
    """

    _mask: torch.Tensor

    def __init__(
        self,
        weight: torch.nn.Parameter,
        bias=None,
        weights_bits=8,
        activation_bits=8,
        sparse_n=4,
        sparse_m=2
    ):
        super(MaskedFakeQuantizedFullyConnectedLayer, self).__init__(
            weight,
            bias=bias,
            weights_bits=weights_bits,
            activation_bits=activation_bits
        )

        self._mask = maskNxM(self.weight.abs(), sparse_n, sparse_m)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight.data = self.weight.data * self._mask

        return super(MaskedFakeQuantizedFullyConnectedLayer, self).forward(input)


class MaskedFakeQuantizedWeightsFullyConnectedLayer(FakeQuantizedFullyConnectedLayer):
    """
    This implementation is used when doing non-ADMM compression of quantization either in combination
    with NxM sparsity (single shot) or predicated on already completing NxM sparsity (double shot).
    """

    _mask: torch.Tensor
    _chunk_size: int

    def __init__(
        self,
        weight: torch.nn.Parameter,
        bias: torch.Tensor = None,
        activation_bits: int = 8,
        weights_bits: int = 8,
        sparse_n: int = 4,
        sparse_m: int = 2,
        chunk_size: int = -1
    ):
        super(MaskedFakeQuantizedWeightsFullyConnectedLayer, self).__init__(
            weight,
            bias=bias,
            weights_bits=weights_bits,
            activation_bits=activation_bits
        )

        self._mask = maskNxM(self.weight.abs(), sparse_n, sparse_m)
        self._chunk_size = chunk_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = SymQuantizer.apply(input, self._activation_bits, 1)

        with torch.no_grad():
            self.weight.data = self.weight.data * self._mask

        quantized_weights = SymQuantizer.apply(self.weight, self._weights_bits, self._chunk_size)

        return F.linear(input, quantized_weights, bias=self.bias)


class FakeQuantizedWeightsFullyConnectedLayer(FakeQuantizedFullyConnectedLayer):

    _chunk_size: int

    def __init__(
        self,
        weight: torch.nn.Parameter,
        bias: torch.Tensor = None,
        activation_bits: int = 8,
        weight_bits: int = 8,
        chunk_size: int = -1
    ):
        super(FakeQuantizedWeightsFullyConnectedLayer, self).__init__(
            weight,
            bias=bias,
            activation_bits=activation_bits,
            weights_bits=weight_bits
        )

        self._chunk_size = chunk_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = SymQuantizer.apply(input, self._activation_bits, 1)
        quantized_weights = SymQuantizer.apply(self.weight, self._weights_bits, self._chunk_size)

        return F.linear(input, quantized_weights, bias=self.bias)
