from typing import Dict, List

from admm_ds.compression_configurations import (
    HFTransformerADMMConfig,
    TransformerComponent
)

from admm_ds.quantized_no_search_admm import SymmetricModifiedQuantizedProjection


def compressSelectedEncoders(
    layers: List[int],
    entry: Dict
) -> HFTransformerADMMConfig:
    config = {}
    for layer in layers:
        key, value, query, atten_output, inter, out = HFTransformerADMMConfig.getAllLayerComponents(layer)
        config[key] = entry
        config[value] = entry
        config[query] = entry
        config[atten_output] = entry
        config[inter] = entry
        config[out] = entry
    return HFTransformerADMMConfig(config)


def compressAttentionOnly(
    entry: Dict
) -> HFTransformerADMMConfig:
    config = {}
    for layer in range(HFTransformerADMMConfig.numLayers()):
        config[HFTransformerADMMConfig.getTransformerModule(layer, TransformerComponent.KEY)] = entry
        config[HFTransformerADMMConfig.getTransformerModule(layer, TransformerComponent.VALUE)] = entry
        config[HFTransformerADMMConfig.getTransformerModule(layer, TransformerComponent.QUERY)] = entry
        config[HFTransformerADMMConfig.getTransformerModule(layer, TransformerComponent.ATTEN_OUTPUT)] = entry
    return HFTransformerADMMConfig(config)


def compressFFOnly(
    entry: Dict
):
    config = {}
    for layer in range(HFTransformerADMMConfig.numLayers()):
        config[HFTransformerADMMConfig.getTransformerModule(layer, TransformerComponent.INTERMEDIATE)] = entry
        config[HFTransformerADMMConfig.getTransformerModule(layer, TransformerComponent.OUTPUT)] = entry
    return HFTransformerADMMConfig(config)


def main() -> None:
    num_bits = 4
    chunk_size = 32

    entry = SymmetricModifiedQuantizedProjection.createConfig(num_bits, chunk_size)

    chosen_encoders = list(range(HFTransformerADMMConfig.numLayers()))
    config = compressSelectedEncoders(chosen_encoders, entry)
    config.saveConfig("./configs/admm_quant_mod/{}_b_{}_cs.json".format(num_bits, chunk_size))


if __name__ == "__main__":
    main()
