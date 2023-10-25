"""
Utilities file to keep track of types of ADMM optimizers we have available
and minor integration utilities.
"""

import enum
from typing import Dict

from .admm_projection import ADMMProjection
from .nxm_admm import (
    NxMProjection,
    NxMQuantizedProjection,
    NxMModifiedQuantizedProjection,
    NxMSTEQuantizedProjection
)
from .quantized_admm import (
    SymmetricQuantizedProjection,
    SymmetricMaskedQuantizedProjection
)
from .quantized_no_search_admm import (
    SymmetricModifiedQuantizedProjection,
    SymmetricMaskedModifiedQuantizedProjection
)
from .admm_disabled_projection import (
    ADMMDisabledNxMCompressor,
    ADMMDisabledMaskedQuantizedCompressor,
    ADMMDisabledSTECompressor
)

ADMM_TYPES: Dict[str, ADMMProjection] = {
    NxMProjection.ProjectionName: NxMProjection,
    SymmetricQuantizedProjection.ProjectionName: SymmetricQuantizedProjection,
    SymmetricModifiedQuantizedProjection.ProjectionName: SymmetricMaskedQuantizedProjection,
    NxMQuantizedProjection.ProjectionName: NxMQuantizedProjection,
    NxMModifiedQuantizedProjection.ProjectionName: NxMModifiedQuantizedProjection,
    SymmetricModifiedQuantizedProjection.ProjectionName: SymmetricModifiedQuantizedProjection,
    SymmetricMaskedModifiedQuantizedProjection.ProjectionName: SymmetricMaskedModifiedQuantizedProjection,
    NxMSTEQuantizedProjection.ProjectionName: NxMSTEQuantizedProjection,
    ADMMDisabledNxMCompressor.ProjectionName: ADMMDisabledNxMCompressor,
    ADMMDisabledMaskedQuantizedCompressor.ProjectionName: ADMMDisabledMaskedQuantizedCompressor,
    ADMMDisabledSTECompressor.ProjectionName: ADMMDisabledSTECompressor
}

ADMM_TYPE_ENUM = enum.Enum("ADMM_TYPE_ENUM", ADMM_TYPES)
