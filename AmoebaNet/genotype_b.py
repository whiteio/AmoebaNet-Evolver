from typing import List

from AmoebaNet.operations import (avg_pool_3x3, conv_1x1, conv_3x3, max_pool_2x2,
                                  max_pool_3x3, none, separable_3x3_2, dil_2_separable_5x5_2)

__all__: List[str] = []

# Genotype of AmoebaNet-B

B_NORMAL_OPERATIONS = [
    (1, conv_1x1),
    (1, max_pool_3x3),
    (1, none),
    (0, separable_3x3_2),
    (1, conv_1x1),
    (0, separable_3x3_2),
    (2, conv_1x1),
    (2, none),
    (1, avg_pool_3x3),
    (5, conv_1x1),    
]

B_NORMAL_CONCAT = [0,3,4,6]

B_REDUCTION_OPERATIONS = [
    (0, max_pool_2x2),
    (0, max_pool_3x3),
    (2, none),
    (1, conv_3x3),
    (2, dil_2_separable_5x5_2),
    (2, max_pool_3x3),
    (3, none),
    (1, separable_3x3_2),
    (4, avg_pool_3x3),
    (3, conv_1x1),
]

B_REDUCTION_CONCAT = [5,6]
