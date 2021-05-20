import numpy as np
import torch
import torch.optim as optim
import AmoebaNet as amoeba
import random

from AmoebaNet.operations import (none,
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

##########################################################################################
##########################################################################################
# About utils.py 
#
# Contains key utility functions that are used by other classes within the system
#
##########################################################################################
##########################################################################################

NUM_CLASSES = 14 # Number of classes in dataset
NUM_NORMAL = 3 # Number of normal cells in model
NUM_FILTERS = 100 # Number of filters in first normal cell will be (100 // 4)**2

# Returns the initial models to add to the population queue where
# the returned model will contain one mutation upon the base model
def get_model():
    print("Getting initial model")
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
   
    ops = [NORMAL_OPERATIONS, REDUCTION_OPERATIONS]

    type_to_mutate = random.randint(0,1)
    op_to_mutate_index = random.randint(0, 9)

    current_op = ops[type_to_mutate][op_to_mutate_index][1]

    ops_tuple_to_list = convert_data_structure_list_tuple(ops[type_to_mutate])

    if random.random() >= 0.95:
        ops_tuple_to_list[op_to_mutate_index][1] = none
    else:
        ops_tuple_to_list[op_to_mutate_index][1] = get_replacement_op(current_op)

    ops[type_to_mutate] = convert_data_structure_list_tuple(ops_tuple_to_list)

    model = amoeba.amoebanet(NUM_CLASSES, NUM_NORMAL, NUM_FILTERS, ops[0], ops[1])

    return model, ops[0], ops[1]

# Returns the optimizer used for the model and learning rate passed in
def get_optimizer(model, LR):
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR, 
        momentum=0.9,
        weight_decay=1e-4)

    return optimizer

# Returns a randomly selected operation from NAS search space
def get_replacement_op(current_op):
    print("Getting replacement op")
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

# Converts a list to tuple and visa versa
def convert_data_structure_list_tuple(t):
    if type(t) == list or type(t) == tuple:
        return [convert_data_structure_list_tuple(i) for i in t]
    return t

# Makes a mutation to a model stored at bot_checkpoint_path, it will be a mutation 
# of the model stored at top_checkpoint_path
def exploit_and_explore(top_checkpoint_path, bot_checkpoint_path):
    print("Exploit and Explore")
    """ Get checkpoint of best model, mutate operations, create new model and save"""
    checkpoint = torch.load(top_checkpoint_path)
    normal_ops = checkpoint['normal_ops']
    reduction_ops = checkpoint['reduction_ops']

    print("Making mutation")
    ops = [normal_ops, reduction_ops]

    type_to_mutate = random.randint(0,1)
    op_to_mutate_index = random.randint(0, 9)

    current_op = ops[type_to_mutate][op_to_mutate_index][1]

    ops_tuple_to_list = convert_data_structure_list_tuple(ops[type_to_mutate])
    
    if random.random() >= 0.95:
        ops_tuple_to_list[op_to_mutate_index][1] = none
    else:
        ops_tuple_to_list[op_to_mutate_index][1] = get_replacement_op(current_op)

    ops[type_to_mutate] = convert_data_structure_list_tuple(ops_tuple_to_list)

    model = amoeba.amoebanet(NUM_CLASSES, NUM_NORMAL, NUM_FILTERS, ops[0], ops[1])
    optimizer = get_optimizer(model, 0.01)

    print(f"Made mutation on device: {torch.cuda.current_device()}")
    checkpoint = dict(model_state_dict=model.state_dict(),
                      optim_state_dict=optimizer.state_dict(),
                      normal_ops=ops[0],
                      reduction_ops=ops[1])
    torch.save(checkpoint, bot_checkpoint_path)
