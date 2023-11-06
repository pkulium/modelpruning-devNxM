import logging
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Dict, List, Optional

import torch

from admm_ds.layers import FakeQuantizedFullyConnectedLayer, MaskedFakeQuantizedFullyConnectedLayer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ADMMProjection(ABC):
    """
    Abstract parent class for all potential constraints in the framework.

    An ADMMProjection instance compresses a particular structure in the model,
    in this case represented by a torch.nn.Module. The simplest case of this
    is a simple fully connected layer (torch.nn.Linear), as long as the module
    is a sub-module of the entire model it will work in this system.
    """

    ProjectionName = "OVERLOAD IN CHILDREN"

    _module: torch.nn.Module
    _name: str
    _u: Optional[torch.Tensor]
    _z: Optional[torch.Tensor]
    _cached_weight: Optional[torch.Tensor]

    @abstractmethod
    def __init__(self, model: torch.nn.Module, targeted_module: str, compression_args: Dict):
        super(ADMMProjection, self).__init__()

        self._module = model.get_submodule(targeted_module)
        self._name = targeted_module
        self._u = None
        self._z = None
        self._cached_weight = None

    @abstractmethod
    def project(self) -> None:
        """
        Project the current state of W and U onto the constraints the ADMMProjection
        instance will optimize for. U should be updated within this method as well.
        """
        pass

    @abstractmethod
    def loss(self) -> torch.Tensor:
        """
        Calculate the Frobenius norm squared as is appropriate for the module.
        """
        pass

    @abstractmethod
    def update_u(self) -> None:
        """
        Update the dual variables owned by the instance according to the following:
        U_k+1 = W_k+1 - Z_k+1 + U_k. Not a global implementation since it's possible
        modules will need to manage multiple copies of each.
        """
        pass

    @abstractmethod
    def prune_module(self) -> None:
        """
        Force the module to meet the compression constraints. ADMM variables are
        persisted in their state across this operation and the parameter is cached to
        its most recent value. Note that doing any operations like project following
        this will update the state of the ADMM variables and make them invalid
        for resumption in the ADMM process.
        """
        pass

    @abstractmethod
    def restore_module(self) -> None:
        """
        Reset the module to its last cached state, checkpointed either at the last
        prune_module call or the ADMMProjection instance's initialization.
        """
        pass

    @abstractmethod
    def get_parameters_for_training(self) -> List[str]:
        """
        Return names of parameters from the module that should be trained.
        """

        return [self._name + "." + name for name, _ in self._module.named_parameters()]

    @abstractmethod
    def match_parameter_device(self) -> None:
        pass

    @abstractstaticmethod
    def compression_ratio(compression_args: Dict) -> float:
        pass

    @property
    def u(self) -> torch.Tensor:
        return self._u

    @property
    def z(self) -> torch.Tensor:
        return self._z


class ADMMFCLProjection(ADMMProjection):
    """
    Partial specialization of ADMMProjection for fully-connected layers.

    This class assumes torch.nn.Linear is the target module, or more broadly,
    that their is a fully-connected layer module with the weight attribute
    pointing to the parameter matrix.
    """

    _u: torch.Tensor
    _z: torch.Tensor
    _cached_weight: torch.Tensor

    @abstractmethod
    def __init__(self, model: torch.nn.Module, targeted_module: str, compression_args: Dict) -> None:
        super(ADMMFCLProjection_Lora, self).__init__(model, targeted_module, compression_args)

        self._cached_weight = self._module.weight.data.detach().cpu()
        # The _like calls are more readable for now, but if/when we move to mixed precision
        # training this assumption will need to be revisited.
        self._u = torch.zeros_like(self._module.weight.data)
        self._z = torch.empty_like(self._module.weight.data)

    def loss(self) -> torch.Tensor:
        return torch.linalg.norm(self._module.weight - (self._z - self._u), "fro") ** 2

    def update_u(self) -> None:
        self._u = self._module.weight.data - self._z + self._u

    def prune_module(self) -> None:
        # Cache ADMM variables, they will be destroyed by the projection
        cached_u = self._u
        cached_z = self._z
        self._u = torch.zeros_like(self._u)

        self.project()

        # Save weight and update actual module
        self._cached_weight = self._module.weight.data
        self._module.weight.data = self._z

        # Restore cached ADMM variables
        self._u = cached_u
        self._z = cached_z

        cached_u = None
        cached_z = None

    def restore_module(self) -> None:
        self._module.weight.data = self._cached_weight

    def get_parameters_for_training(self) -> List[torch.Tensor]:
        return super().get_parameters_for_training()

    def match_parameter_device(self):
        if self._u.device != self._module.weight.device:
            self._u = self._u.to(device=self._module.weight.device)
        if self._z.device != self._module.weight.device:
            self._z = self._z.to(device=self._module.weight.device)
        assert(self._z.device == self._module.weight.device)
        assert(self._u.device == self._module.weight.device)


class ADMMFCLProjection_Lora(ADMMProjection):
    """
    Partial specialization of ADMMProjection for fully-connected layers.

    This class assumes torch.nn.Linear is the target module, or more broadly,
    that their is a fully-connected layer module with the weight attribute
    pointing to the parameter matrix.
    """

    _u: torch.Tensor
    _z: torch.Tensor
    _cached_weight: torch.Tensor

    @abstractmethod
    def __init__(self, model: torch.nn.Module, targeted_module: str, compression_args: Dict) -> None:
        super(ADMMFCLProjection_Lora, self).__init__(model, targeted_module, compression_args)

        self._cached_weight = self._module.weight.data.detach().cpu()
        # The _like calls are more readable for now, but if/when we move to mixed precision
        # training this assumption will need to be revisited.
        self._u = torch.zeros_like(self._module.weight.data)
        self._z = torch.empty_like(self._module.weight.data)

    def loss(self) -> torch.Tensor:
        return torch.linalg.norm(self._module.weight + self._module.lora_B.weight @ self._module.lora_A.weight - (self._z - self._u), "fro") ** 2

    def update_u(self) -> None:
        self._u = self._module.weight.data + self._module.lora_B.weight.data @ self._module.lora_A.weight.data - self._z + self._u

    def prune_module(self) -> None:
        # Cache ADMM variables, they will be destroyed by the projection
        cached_u = self._u
        cached_z = self._z
        self._u = torch.zeros_like(self._u)

        self.project()

        # Save weight and update actual module
        self._cached_weight = self._module.weight
        self._module.weight.data = self._z
        self._module.disabled = True

        # Restore cached ADMM variables
        self._u = cached_u
        self._z = cached_z

        cached_u = None
        cached_z = None

    def restore_module(self) -> None:
        self._module.weight.data = self._cached_weight
        self._module.disabled = False

    def get_parameters_for_training(self) -> List[torch.Tensor]:
        return super().get_parameters_for_training()

    def match_parameter_device(self):
        if self._u.device != self._module.weight.device:
            self._u = self._u.to(device=self._module.weight.device)
        if self._z.device != self._module.weight.device:
            self._z = self._z.to(device=self._module.weight.device)
        assert(self._z.device == self._module.weight.device)
        assert(self._u.device == self._module.weight.device)

class ADMMQuantizedFCLProjection(ADMMFCLProjection):
    """
    Partial specialization of ADMMFCLProjection to handle when the compression scheme needs
    to quantize the activations. This is performed by module replacement with the
    FakeQuantizedFullyConnectedLayer module which performs fake quantization on the
    inputs to the linear layer.

    NOTE: Since this initializer is called before the loading of the compression parameters,
    it is necessary to set self._module.weight_bits (for debug purposes) or self._module.activation_bits
    (for changing from the default 8) to the appropriate number in the child class' __init__ method.
    """

    _module: FakeQuantizedFullyConnectedLayer

    @abstractmethod
    def __init__(self, model: torch.nn.Module, targeted_module: str, compression_args: Dict) -> None:
        super(ADMMQuantizedFCLProjection, self).__init__(model, targeted_module, compression_args)

        # We assume that the entire module is not encapsulated by our single module. This is a valid
        # assumption for our uses cases, but isn't fully general
        parent_module_name = self._name[:self._name.rfind(".")]
        attribute_name = self._name[self._name.rfind(".")+1:]

        # Replace the Linear layer with an FCL that does fake quantization instead
        parent_module = model.get_submodule(parent_module_name)
        new_module = FakeQuantizedFullyConnectedLayer(self._module.weight, bias=self._module.bias)
        setattr(parent_module, attribute_name, new_module)

        # Update our reference to the new submobule in the model
        self._module = model.get_submodule(self._name)


class ADMMMaskedQuantizedFCLProjection(ADMMFCLProjection):

    _module: MaskedFakeQuantizedFullyConnectedLayer

    def __init__(self, model: torch.nn.Module, targeted_module: str, compression_args: Dict) -> None:
        super(ADMMMaskedQuantizedFCLProjection, self).__init__(model, targeted_module, compression_args)

        # We assume that the entire module is not encapsulated by our single module. This is a valid
        # assumption for our uses cases, but isn't fully general
        parent_module_name = self._name[:self._name.rfind(".")]
        attribute_name = self._name[self._name.rfind(".")+1:]

        # Replace the Linear layer with a masked FCL quantized layer
        parent_module = model.get_submodule(parent_module_name)
        new_module = MaskedFakeQuantizedFullyConnectedLayer(self._module.weight, bias=self._module.bias)
        setattr(parent_module, attribute_name, new_module)

        # Update our reference to get the new submodule in the model.
        self._module = model.get_submodule(self._name)
