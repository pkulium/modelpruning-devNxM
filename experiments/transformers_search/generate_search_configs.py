from admm_ds.compression_configurations import SearchExperimentConfig, TransformerComponent
from admm_ds.nxm_admm import NxMProjection, NxMModifiedQuantizedProjection
from admm_ds.quantized_no_search_admm import SymmetricModifiedQuantizedProjection


def main() -> None:

    compression_configs = [
        NxMProjection.createConfig(4, 2),
        SymmetricModifiedQuantizedProjection.createConfig(8, 32),
        SymmetricModifiedQuantizedProjection.createConfig(4, 32),
        NxMModifiedQuantizedProjection.createConfig(4, 2, 8, 32),
        NxMModifiedQuantizedProjection.createConfig(4, 2, 4, 32)
    ]

    config = {
        TransformerComponent.KEY: compression_configs,
        TransformerComponent.QUERY: compression_configs,
        TransformerComponent.VALUE: compression_configs,
        TransformerComponent.ATTEN_OUTPUT: compression_configs,
        TransformerComponent.INTERMEDIATE: compression_configs,
        TransformerComponent.OUTPUT: compression_configs
    }

    full_config = {
        SearchExperimentConfig.COMPRESSION_KEY: config,
        SearchExperimentConfig.HIDDEN_SIZE_KEY: 768,
        SearchExperimentConfig.INTERMEDIATE_SIZE_KEY: 3072
    }

    formal_config = SearchExperimentConfig(full_config)
    formal_config.saveConfig("configs/base_config.json")


if __name__ == "__main__":
    main()
