from admm_ds.compression_configurations import SearchExperimentConfig

config_file = "../configs/base_config.json"
loaded_configuration = SearchExperimentConfig(config_file)

count = 0
for config_id, config in loaded_configuration:
    count += 1

assert(count == 5 ** 6)

for confid_id, config in loaded_configuration:
    print(config._configuration)
    break
