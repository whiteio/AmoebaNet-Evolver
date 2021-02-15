import numpy as np
import torch
import torch.optim as optim
import AmoebaNetAll as amoeba
import random

from AmoebaNetAll.operations import (none,
        avg_pool_3x3,
        max_pool_3x3,
        max_pool_2x2,
        conv_1x7_7x1,
        conv_1x1,
        conv_3x3,
        separable_7x7_2,
        separable_3x3_2,
        separable_5x5_2,
        dil_2_separable_5x5_2)

NUM_CLASSES = 14
NUM_NORMAL = 3
NUM_FILTERS = 100

def get_model():
    # IMPORTANT! - FIRST OPERATION SHOULD BE CONV_1X1
    """Returns model with random mutation to a single op"""
    # Create method in amoeba that randomly mutates an op
    NORMAL_OPERATIONS = [
        (1, conv_1x1),
        (1, max_pool_3x3),
        (1, none),
        (0, conv_1x7_7x1),
        (0, conv_1x1), 
        (0, conv_1x7_7x1),
        (2, max_pool_3x3),
        (2, none),
        (1, avg_pool_3x3),
        (5, conv_1x1),
    ]

    REDUCTION_OPERATIONS = [
        (0, max_pool_2x2),
        (0, max_pool_3x3),
        (2, none),
        (1, conv_3x3),
        (2, conv_1x7_7x1),
        (2, max_pool_3x3),
        (3, none),
        (1, max_pool_2x2),
        (2, avg_pool_3x3),
        (3, conv_1x1),
    ]

    model = amoeba.amoebanet(NUM_CLASSES, NUM_NORMAL, NUM_FILTERS, NORMAL_OPERATIONS, REDUCTION_OPERATIONS)

    return model, NORMAL_OPERATIONS, REDUCTION_OPERATIONS

def get_optimizer(model, LR):
    """Helper method to get optimizer for model with lr=LR"""
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR, 
        momentum=0.9,
        weight_decay=1e-4)

    return optimizer

def get_replacement_op(current_op):
    nas_space = [
        avg_pool_3x3,
        max_pool_3x3,
        max_pool_2x2,
        conv_1x7_7x1,
        conv_1x1,
        conv_3x3,
        separable_7x7_2,
        separable_3x3_2,
        separable_5x5_2,
        dil_2_separable_5x5_2
    ]

    new_op_index = random.randint(0, 9)

    if nas_space[new_op_index] == current_op:
        return get_replacement_op(current_op)
    else:
        return nas_space[new_op_index]

def f(t):
    if type(t) == list or type(t) == tuple:
        return [f(i) for i in t]
    return t

# ONLY COPY THE MUTATION FROM THE MODEL BEING EXPLORED
def exploit_and_explore(top_checkpoint_path, bot_checkpoint_path):
    """ Get checkpoint of best model, mutate operations, create new model and save"""
    checkpoint = torch.load(top_checkpoint_path)
    normal_ops = checkpoint['normal_ops']
    reduction_ops = checkpoint['reduction_ops']

    ops = [normal_ops, reduction_ops]

    type_to_mutate = random.randint(0,1)
    op_to_mutate_index = random.randint(0, 9)

    current_op = ops[type_to_mutate][op_to_mutate_index][1]

    ops_tuple_to_list = f(ops[type_to_mutate])
    
    if random.random() >= 0.95:
        ops_tuple_to_list[op_to_mutate_index][1] = none
    else:
        ops_tuple_to_list[op_to_mutate_index][1] = get_replacement_op(current_op)

    ops[type_to_mutate] = f(ops_tuple_to_list)

    model = amoeba.amoebanet(NUM_CLASSES, NUM_NORMAL, NUM_FILTERS, ops[0], ops[1])
    optimizer = get_optimizer(model, 0.01)
    print(f"Made mutation on device: {torch.cuda.current_device()}")
    checkpoint = dict(model_state_dict=model.state_dict(),
                      optim_state_dict=optimizer.state_dict(),
                      normal_ops=ops[0],
                      reduction_ops=ops[1])
    torch.save(checkpoint, bot_checkpoint_path)


