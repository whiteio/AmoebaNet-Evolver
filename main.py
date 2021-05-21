import argparse
import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as _mp
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from trainer import Trainer
from utils import get_optimizer, exploit_and_explore, get_model
from random import randrange

from random import randint

mp = _mp.get_context('spawn')

##########################################################################################
##########################################################################################
# About main.py
#
# 
# Sample command to run the system:
# `$ python main.py --device cuda --population_size 10`
#
# Worker process is responsible from taking a model from the population queue,
# training, evaluating and then placing the model on the finished queue and 
# repeating this process until the correct number of mutations have occurred.
#
#
# Explorer process is responsible for taking models off the finished queue, 
# making a mutation to them and placing them back into the population queue.
#
# Note: There is no communication between processes to exchange model details, this
# is all managed by the two queues to store the models information, the models are saved 
# in a directory and they can be loaded by using 'torch.load(model_path...)'

##########################################################################################
##########################################################################################


# Responsible for getting models from queue, and using trainer class
# to put models on the finished queue
class Worker(mp.Process):
    def __init__(self, mutation_count, mutation_search_max_count, population, finish_tasks,
                 device, data_path):
        super().__init__()
        self.mutation_count = mutation_count
        self.population = population
        self.finish_tasks = finish_tasks
        self.mutation_search_max_count = mutation_search_max_count
        self.device = device
        model, normal_ops, reduction_ops = get_model()
        model = model.to(device)
        optimizer = get_optimizer(model, 0.01)
        self.trainer = Trainer(model=model,
                               normal_ops=normal_ops,
                               reduction_ops=reduction_ops,
                               optimizer=optimizer,
                               data_path=data_path,
                               loss_fn=nn.BCEWithLogitsLoss(),
                               device=self.device)

    def run(self):
        while True:
            if self.mutation_count.value > self.mutation_search_max_count:
                break

            # Get a model from population queue
            task = self.population.get()
            self.trainer.set_id(task['id'])
            checkpoint_path = "checkpoints/task-%03d.pth" % task['id']
            if os.path.isfile(checkpoint_path):
                # Load model from path
                self.trainer.load_checkpoint(checkpoint_path)
            try:
                self.trainer.train() # Train model
                score = self.trainer.eval() # Evaluate model 
                self.trainer.save_checkpoint(checkpoint_path) # Save model and score
                self.finish_tasks.put(dict(id=task['id'], score=score)) # Place on finished queue
            except KeyboardInterrupt:
                break


class Explorer(mp.Process):
    def __init__(self, mutation_count, mutation_search_max_count, population, finish_tasks):
        super().__init__()
        self.mutation_count = mutation_count
        self.population = population
        self.finish_tasks = finish_tasks
        self.mutation_search_max_count = mutation_search_max_count

    def run(self):
        while True:
            if self.mutation_count.value > self.mutation_search_max_count:
                print("Reached mutation cout")
                break
            if self.population.empty() and self.finish_tasks.full():
                print("Exploit and explore")
                tasks = []
                
                # Create list of models by removing all from finished queue
                while not self.finish_tasks.empty():
                    tasks.append(self.finish_tasks.get())

                # Sort in descending order based on average AUC for each model
                tasks = sorted(tasks, key=lambda x: x['score'], reverse=True)
                
                print('Best score on', tasks[0]['id'], 'is', tasks[0]['score'])
                print('Worst score on', tasks[-1]['id'], 'is', tasks[-1]['score'])
                
                # Top 50% in the case where theres only 4 models in population, 
                # would be reduced to 20% if there was a larger population size
                fraction = 0.50
                cutoff = int(np.ceil(fraction * len(tasks)))
                
                tops = tasks[:cutoff]

                for model in reversed(tasks):
                    if model not in tops:
                        random_top_model = tops[randrange(len(tops))-1]
                        top_checkpoint_path = "checkpoints/task-%03d.pth" % random_top_model['id']
                    else:
                        top_checkpoint_path = "checkpoints/task-%03d.pth" % model['id']

                    bot_checkpoint_path = "checkpoints/task-%03d.pth" % model['id']

                    exploit_and_explore(top_checkpoint_path, bot_checkpoint_path)

                    with self.mutation_count.get_lock():
                        self.mutation_count.value += 1

                    self.population.put(model)

                    print('Models from previous cycle mutated and replaced population')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--device", type=str, default='cuda:', help="")
    parser.add_argument("--population_size", type=int, default=10, help="")
    parser.add_argument("--data_path", type=str, deafult='')
    parser.add_argument("--test_mode", type=bool, deafult=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mutation_count", type=int, default=30)
    parser.add_argument("--gpu-count", type=int, default=1)

    args = parser.parse_args()
    mp = mp.get_context('forkserver')

    # Set device
    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = args.device

    data_path = args.data_path
    population_size = args.population_size
    batch_size = args.batch_size
    mutation_search_max_count = args.mutation_count
    gpu_count = args.gpu_count

    # Create directory to store model checkpoints
    pathlib.Path('checkpoints').mkdir(exist_ok=True)
    checkpoint_str = "checkpoints/task-%03d.pth"

    population = mp.Queue(maxsize=population_size)
    finish_tasks = mp.Queue(maxsize=population_size)

    mutation_count = mp.Value('i', 0)
    for i in range(population_size):
        population.put(dict(id=i, score=0))

    workers = []

    print("Create Workers")
    for i in range(0, gpu_count):
        workers.append(Worker(mutation_count, mutation_search_max_count, population, finish_tasks, f"{device}{i}", data_path))
  
    print("Create Explorer")
    workers.append(Explorer(mutation_count, mutation_search_max_count, population, finish_tasks))

    [w.start() for w in workers]
    [w.join() for w in workers]

    task = []
    while not finish_tasks.empty():
        task.append(finish_tasks.get())
    while not population.empty():
        task.append(population.get())
    task = sorted(task, key=lambda x: x['score'], reverse=True)
    print('Best score in last run: ', task[0]['id'], 'is', task[0]['score'])
