from typing import List

import torch
from transformers import AutoModelForSequenceClassification

from admm_ds.compression_configurations import HFTransformerADMMConfig
from admm_ds.admm_optimizer import ADMMOptimizer


def main() -> None:
    config = HFTransformerADMMConfig.uniformFusedNxMQuantized()
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    model.to("cuda:0")
    optim = ADMMOptimizer(model, config, rho=1e-2)
    model.train()
    optim.regularizer()
    print(optim._instances[0]._z)
    print(optim._instances[0]._u)

    params: List[str] = optim.get_parameters_for_training()
    first_param: torch.nn.parameter.Parameter = model.get_parameter(params[0])
    print(first_param)
    print(first_param.grad)


if __name__ == "__main__":
    main()
