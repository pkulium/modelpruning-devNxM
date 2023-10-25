import torch

from admm_ds.quantization_utils import quantizeTensor

out_neurons = 768
in_neurons = 768
scale = 0.0087

layer = torch.nn.Linear(in_neurons, out_neurons)

for name, parameter in layer.named_parameters():
    if name == "weight":

        print(parameter.data.size())

        ourQuantization = quantizeTensor(parameter.data, scale, 0, 8)
        torchQuantization = torch.fake_quantize_per_tensor_affine(parameter.data, scale, 0, -128, 127)

        print((ourQuantization - torchQuantization).abs().max())
