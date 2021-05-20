from utils import *
import pytest

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

def test_get_model():
	model = get_model()
	assert model is not None
	
	diff_count = 0

	i = 0
	while i < len(NORMAL_OPERATIONS):
		if model[1][i][1] != NORMAL_OPERATIONS[i][1]:
			diff_count += 1
		i += 1

	while i < len(REDUCTION_OPERATIONS):
		if model[2][i][1] != REDUCTION_OPERATIONS[i][1]:
			diff_count += 1
		i += 1

	assert diff_count <= 1

def test_get_optimizer():
	model = get_model()[0]
	lr = 0.01
	optimizer = get_optimizer(model, lr)
	assert optimizer is not None

def test_get_new_operation():
	current_op = avg_pool_3x3
	assert get_replacement_op(current_op) is not None


def test_get_new_operation_no_duplicate():
	current_op = avg_pool_3x3
	for i in range(0,100):
		assert current_op is not get_replacement_op(current_op)

def test_exploit_and_explore():
	print("")
	test_model = get_model()[0]
	test_optim = get_optimizer(test_model, 0.01)

	checkpoint = dict(model_state_dict=test_model.state_dict(),
							optim_state_dict=test_optim.state_dict(),
							normal_ops=NORMAL_OPERATIONS,
							reduction_ops=REDUCTION_OPERATIONS)
	torch.save(checkpoint, "checkpoints/tester")

	checkpoint = dict(model_state_dict=test_model.state_dict(),
							optim_state_dict=test_optim.state_dict(),
							normal_ops=NORMAL_OPERATIONS,
							reduction_ops=REDUCTION_OPERATIONS)
	torch.save(checkpoint, "checkpoints/tester1")

	if torch.cuda.is_available():
		exploit_and_explore("checkpoints/tester", "checkpoints/tester1")

	temp_model = torch.load("checkpoints/tester1")
	normal_ops = temp_model['normal_ops']
	reduction_ops = temp_model['reduction_ops']

	diff_count = 0

	i = 0
	while i < len(NORMAL_OPERATIONS):
		if normal_ops[i][1] != NORMAL_OPERATIONS[i][1]:
			diff_count += 1
		i += 1

	while i < len(REDUCTION_OPERATIONS):
		if reduction_ops[i][1] != REDUCTION_OPERATIONS[i][1]:
			diff_count += 1
		i += 1

	assert diff_count <= 1


def test_exploit_and_explore_invalid_checkpoint():
	with pytest.raises(Exception) as E:
		exploit_and_explore("aaaa","aaaaaaa")

def test_convert_tuple_to_list():
	assert type(convert_data_structure_list_tuple(convert_data_structure_list_tuple(NORMAL_OPERATIONS))) == type(NORMAL_OPERATIONS)

