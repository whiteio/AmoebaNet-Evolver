from typing import List

from AmoebaNetAll.operations import (avg_pool_3x3, max_pool_2x2, max_pool_3x3, 
                                    none, separable_3x3_2, separable_7x7_2, separable_5x5_2)

__all__: List[str] = []

# The genotype for AmoebaNet-D
C_NORMAL_OPERATIONS = [
    (0, avg_pool_3x3),
    (0, separable_3x3_2),
    (0, none),
    (0, separable_3x3_2),
    (2, avg_pool_3x3),
    (1, separable_3x3_2),
    (0, none),
    (1, separable_3x3_2),
    (3, avg_pool_3x3),
    (0, separable_3x3_2),
]


C_NORMAL_CONCAT = [1, 2, 4, 5, 6]

C_REDUCTION_OPERATIONS = [
    (0, max_pool_3x3),
    (0, max_pool_3x3),
    (2, separable_7x7_2),
    (0, separable_3x3_2),
    (0, separable_7x7_2),
    (1, max_pool_3x3),
    (4, separable_5x5_2),
    (4, separable_5x5_2),
    (1, max_pool_3x3),
    (1, separable_3x3_2)
]

C_REDUCTION_CONCAT = [0, 2, 3, 4, 5, 6]
