import torch

from admm_ds.quantization_utils import quantizeSymmetricSimple

sequence_length = 4
batch_size = 2
token_size = 6

activation = torch.rand((batch_size, sequence_length, token_size))

quantizedActivations = quantizeSymmetricSimple(activation, 4, 1)
print(quantizedActivations)
