from typing import List

from AmoebaNet.operations import (avg_pool_3x3, conv_1x7_7x1, max_pool_3x3, none,
                                  separable_3x3_2, separable_5x5_2, separable_7x7_2)

__all__: List[str] = []

# The genotype for AmoebaNet-A
A_NORMAL_OPERATIONS = [
    (0, avg_pool_3x3),
    (0, max_pool_3x3),
    (1, separable_3x3_2),
    (1, none),
    (0, none),
    (1, avg_pool_3x3),
    (0, separable_3x3_2),
    (2, separable_5x5_2),
    (5, avg_pool_3x3),
    (0, separable_3x3_2),
]


A_NORMAL_CONCAT = [1, 3, 4, 6]

A_REDUCTION_OPERATIONS = [
    (1, separable_3x3_2),
    (0, avg_pool_3x3),
    (0, max_pool_3x3),
    (2, separable_7x7_2),
    (1, max_pool_3x3),
    (0, max_pool_3x3),
    (4, separable_3x3_2),
    (0, conv_1x7_7x1),
    (1, avg_pool_3x3),
    (0, separable_7x7_2),
]

A_REDUCTION_CONCAT = [2, 3, 4, 5, 6]
