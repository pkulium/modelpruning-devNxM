import itertools
import json
from abc import ABC, abstractclassmethod
from enum import Enum, unique
from os import path
from typing import Dict, Tuple, Union

from admm_ds.admm_types import ADMM_TYPES

from .admm_projection import ADMMProjection

IMPORT_TYPE_KEY = "type"
ARGUMENTS_KEY = "args"


@unique
class TransformerComponent(Enum):
    KEY = "key"
    VALUE = "value"
    QUERY = "query"
    ATTEN_OUTPUT = "attention_output"
    INTERMEDIATE = "intermediate"
    OUTPUT = "output"
    ATTENTION = "attention"


class TransformerADMMConfig(ABC):
    _configuration = Dict

    @classmethod
    def buildEntry(cls, projection_class: ADMMProjection, class_arguments: Dict) -> Dict:
        projection_name = projection_class.ProjectionName
        return {
            IMPORT_TYPE_KEY: projection_name,
            ARGUMENTS_KEY: class_arguments
        }

    @classmethod
    def getAllLayerComponents(cls, layer: int) -> Tuple[str, str, str, str, str, str]:
        key = cls.getTransformerModule(layer, TransformerComponent.KEY)
        value = cls.getTransformerModule(layer, TransformerComponent.VALUE)
        query = cls.getTransformerModule(layer, TransformerComponent.QUERY)
        atten_output = cls.getTransformerModule(layer, TransformerComponent.ATTEN_OUTPUT)
        inter = cls.getTransformerModule(layer, TransformerComponent.INTERMEDIATE)
        out = cls.getTransformerModule(layer, TransformerComponent.OUTPUT)
        return key, value, query, atten_output, inter, out

    @abstractclassmethod
    def numLayers(cls):
        pass

    @abstractclassmethod
    def getTransformerModule(cls, layer: int, type: TransformerComponent) -> str:
        """
        Returns the module name for the specific transformer component requested.
        """
        pass

    def __init__(self, configuration: Dict) -> None:
        self._configuration = configuration

    def saveConfig(self, file_name: str) -> None:
        with open(file_name, "w") as output_stream:
            json.dump(self._configuration, output_stream, indent=4)

    @property
    def configuration(self):
        return self._configuration

    def __repr__(self) -> str:
        return repr(self._configuration)


class HFTransformerADMMConfig(TransformerADMMConfig):

    @classmethod
    def numLayers(cls):
        return 12

    @classmethod
    def getTransformerModule(cls, layer: int, type: TransformerComponent) -> str:
        base_string = "bert.encoder.layer.{}.".format(layer)
        if type == TransformerComponent.KEY:
            return base_string + "attention.self.key"
        elif type == TransformerComponent.VALUE:
            return base_string + "attention.self.value"
        elif type == TransformerComponent.QUERY:
            return base_string + "attention.self.query"
        elif type == TransformerComponent.ATTEN_OUTPUT:
            return base_string + "attention.output.dense"
        elif type == TransformerComponent.INTERMEDIATE:
            return base_string + "intermediate.dense"
        elif type == TransformerComponent.OUTPUT:
            return base_string + "output.dense"
        elif type == TransformerComponent.ATTENTION:
            return base_string + "attention"
        else:
            raise ValueError("Invalid type: {}. Not an attribute of TransformerComponent".format(
                type
            ))

    def compressionRatio(self, hidden_size: int, intermediate_size: int) -> float:
        original_size = 0
        compressed_size = 0

        for name, compression in self._configuration.items():
            layer_compression = ADMM_TYPES[compression[IMPORT_TYPE_KEY]].compression_ratio(
                compression[ARGUMENTS_KEY]
            )

            if (
                name.find("attention.self.key") != -1 or
                name.find("attention.self.value") != -1 or
                name.find("attention.self.query") != -1 or
                name.find("attention.output.dense") != -1
            ):
                layer_size = hidden_size * hidden_size * 4
            else:
                layer_size = hidden_size * intermediate_size * 4

            original_size += layer_size
            compressed_size += layer_size * layer_compression

        return compressed_size / original_size


class SearchExperimentConfig:
    """
    Config structure:
    {
        COMPRESSION_KEY: {
            KEY: {
                Each compression technique
            },
            ...
            ...
            ...
        },
        HIDDEN_SIZE_KEY: 768,
        INTERMEDIATE_SIZE_KEY: 3072
    }
    """

    _configuration: Dict
    _iter_product: itertools.product
    _config_id: int
    IDX_KEY = 0
    IDX_QUERY = 1
    IDX_VALUE = 2
    IDX_ATTEN_OUTPUT = 3
    IDX_INTERMEDIATE = 4
    IDX_OUTPUT = 5

    COMPRESSION_KEY = "compression_types"
    HIDDEN_SIZE_KEY = "hidden"
    INTERMEDIATE_SIZE_KEY = "intermediate"
    COMPRESSION_TARGET_KEY = "target_ratio"

    def __init__(self, config: Union[str, Dict]):

        if type(config) is dict:
            self._configuration = config
        elif type(config) is str:
            if path.exists(config):
                with open(config, "r") as configuration_file:
                    self._configuration = json.load(configuration_file)
                    compression_configs = self._configuration[SearchExperimentConfig.COMPRESSION_KEY]
                    self._configuration[SearchExperimentConfig.COMPRESSION_KEY] = {
                        TransformerComponent(key): value for key, value in compression_configs.items()
                    }
            else:
                raise ValueError("Configuration file path {} does not exist.".format(config))
        else:
            raise TypeError("config is of type {} but must be either dict or string.".format(type(config)))

        compression_configs = self._configuration[SearchExperimentConfig.COMPRESSION_KEY]

        if (
            TransformerComponent.KEY not in compression_configs or
            TransformerComponent.QUERY not in compression_configs or
            TransformerComponent.VALUE not in compression_configs or
            TransformerComponent.ATTEN_OUTPUT not in compression_configs or
            TransformerComponent.INTERMEDIATE not in compression_configs or
            TransformerComponent.OUTPUT not in compression_configs
        ):
            raise ValueError("Configuration file does not have all Transformer components")

    def __iter__(self):
        compression_configs = self._configuration[SearchExperimentConfig.COMPRESSION_KEY]

        self._iter_product = itertools.product(
            compression_configs[TransformerComponent.KEY],
            compression_configs[TransformerComponent.QUERY],
            compression_configs[TransformerComponent.VALUE],
            compression_configs[TransformerComponent.ATTEN_OUTPUT],
            compression_configs[TransformerComponent.INTERMEDIATE],
            compression_configs[TransformerComponent.OUTPUT]
        )
        self._config_id = 0
        return self

    def __next__(self) -> Tuple[int, Dict]:
        config_id = self._config_id
        self._config_id += 1
        raw_config = next(self._iter_product)

        complete_config = {}
        for layer_index in range(HFTransformerADMMConfig.numLayers()):
            components = HFTransformerADMMConfig.getAllLayerComponents(layer_index)
            key, value, query, atten_output, intermediate, output = components
            complete_config[key] = raw_config[SearchExperimentConfig.IDX_KEY]
            complete_config[value] = raw_config[SearchExperimentConfig.IDX_VALUE]
            complete_config[query] = raw_config[SearchExperimentConfig.IDX_QUERY]
            complete_config[atten_output] = raw_config[SearchExperimentConfig.IDX_ATTEN_OUTPUT]
            complete_config[intermediate] = raw_config[SearchExperimentConfig.IDX_INTERMEDIATE]
            complete_config[output] = raw_config[SearchExperimentConfig.IDX_OUTPUT]

        return (config_id, HFTransformerADMMConfig(complete_config))

    def modelDimensions(self):
        return (
            self._configuration[SearchExperimentConfig.HIDDEN_SIZE_KEY],
            self._configuration[SearchExperimentConfig.INTERMEDIATE_SIZE_KEY]
        )

    def compressionTarget(self):
        return self._configuration[SearchExperimentConfig.COMPRESSION_TARGET_KEY]

    def saveConfig(self, file_name: str) -> None:
        with open(file_name, "w") as output_stream:
            saveable_representation = self._configuration

            saveable_representation[SearchExperimentConfig.COMPRESSION_KEY] = {
                key.value: value for key, value in self._configuration[SearchExperimentConfig.COMPRESSION_KEY].items()
            }
            json.dump(saveable_representation, output_stream, indent=4)

    def __repr__(self) -> str:
        hidden, intermediate = self.modelDimensions()
        return "Target ratio: {}\nHidden: {}\nIntermediate: {}\nConfig: {}".format(
            self._configuration[SearchExperimentConfig.COMPRESSION_TARGET_KEY],
            hidden,
            intermediate,
            self._configuration[SearchExperimentConfig.COMPRESSION_KEY]
        )
