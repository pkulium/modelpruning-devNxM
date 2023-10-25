import torch

from admm_ds.quantization_utils import fixedOffsetQuantizationSearch, quantizeSymmetricSimple

out_neurons = 8
in_neurons = 4

raw_data = [
    [1., 2., 4., 7.],
    [1., 2., 4., -7.],
    [1., 2., 4., 16.],
    [-7., 3., 3., 7.],
    [1., 1., 1., 1.],
    [1., 1., 1., 64.],
    [10., 1., 7., 8.],
    [1., 2.5, 3., 8.]
]

data_tensor = torch.tensor(raw_data)
print(data_tensor)

print(fixedOffsetQuantizationSearch(data_tensor, 4, 0, 1, 2))
print(fixedOffsetQuantizationSearch(data_tensor, 4, 0, 2, 2))
print(fixedOffsetQuantizationSearch(data_tensor, 4, 0, 4, 2))
print(fixedOffsetQuantizationSearch(data_tensor, 4, 0, 8, 2))

print(quantizeSymmetricSimple(data_tensor, 4, 1))
print(quantizeSymmetricSimple(data_tensor, 4, 2))
print(quantizeSymmetricSimple(data_tensor, 4, 4))
print(quantizeSymmetricSimple(data_tensor, 4, 8))
