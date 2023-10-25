from time import perf_counter
from typing import Optional

import torch

import admm_ds.quantization_utils as qu
import admm_ds.nxm_utils as nu


class TrivialProfiler:

    _begin: float
    _end: Optional[float]

    def __init__(self):
        self._begin = perf_counter()
        self._end = None

    def end(self) -> None:
        self._end = perf_counter()

    def duration(self) -> float:
        assert(self._end is not None)
        return self._end - self._begin


num_samples = 100
input_shape = (768, 768)
do_print = False
do_mask = True

total_simple_fitness = 0.0
total_search_fitness = 0.0
time_simple = 0.0
time_search = 0.0

for iter in range(num_samples):
    raw_tensor = torch.rand(input_shape).cuda()

    if do_mask:
        mask = nu.maskNxM(raw_tensor.abs(), 4, 2)
        raw_tensor = raw_tensor * mask

    sym_time = TrivialProfiler()
    sym_quantization = qu.quantizeSymmetricSimple(raw_tensor, 4, 16)
    sym_time.end()
    time_simple += sym_time.duration()
    simple_fitness = qu.evaluateQuantization(sym_quantization.flatten(), raw_tensor.flatten())
    total_simple_fitness += simple_fitness
    if do_print is True:
        print("Symmetric Simple")
        print(simple_fitness)
        print(sym_quantization)

    search_time = TrivialProfiler()
    scales, search_quantization = qu.fixedOffsetQuantizationSearch(
        raw_tensor, 4, 0, 16, 2
    )
    search_time.end()
    time_search += search_time.duration()

    search_fitness = qu.evaluateQuantization(search_quantization.flatten(), raw_tensor.flatten())
    total_search_fitness += search_fitness
    if do_print is True:
        print("Search quantization")
        print(search_fitness)
        print(search_quantization)

    if simple_fitness < search_fitness:
        print("Failed assumption on iteration {}: {} < {}".format(iter + 1, simple_fitness, search_fitness))

print("Simple fitness: {}".format(total_simple_fitness))
print("Simple average time: {} ms".format(time_simple / num_samples * 1000))
print("Search fitness: {}".format(total_search_fitness))
print("Search average time: {} ms".format(time_search / num_samples * 1000))
