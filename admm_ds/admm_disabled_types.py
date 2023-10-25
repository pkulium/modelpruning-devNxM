"""
Analog to admm_types.py. Used to silo the information used for non-ADMM
compression in the framework.
"""

import enum
from typing import Dict

from .admm_disabled_projection import (
    ADMMDisabledProjection,
    ADMMDisabledNxMCompressor,
    ADMMDisabledMaskedQuantizedCompressor,
    ADMMDisabledSTECompressor
)


ADMM_DISABLED_TYPES: Dict[str, ADMMDisabledProjection] = {
    ADMMDisabledNxMCompressor.ProjectionName: ADMMDisabledNxMCompressor,
    ADMMDisabledMaskedQuantizedCompressor.ProjectionName: ADMMDisabledMaskedQuantizedCompressor,
    ADMMDisabledSTECompressor.ProjectionName: ADMMDisabledSTECompressor
}


ADMM_DISABLED_TYPE_ENUM = enum.Enum("ADMM_DISABLED_TYPE_ENUM", ADMM_DISABLED_TYPES)
