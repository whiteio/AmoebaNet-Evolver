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

from random import randint

mp = _mp.get_context('spawn')

# `$ python main.py --device cuda --population_size 10`


class Worker(mp.Process):
    def __init__(self, mutation_count, mutation_search_max_count, population, finish_tasks,
                 device):
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
                               loss_fn=nn.BCEWithLogitsLoss(),
                               device=self.device)

    def run(self):
        while True:
            if self.mutation_count.value > self.mutation_search_max_count:
                break
            # Train
            task = self.population.get()
            self.trainer.set_id(task['id'])
            checkpoint_path = "checkpoints/task-%03d.pth" % task['id']
            if os.path.isfile(checkpoint_path):
                self.trainer.load_checkpoint(checkpoint_path)
            try:
                self.trainer.train()
                score = self.trainer.eval()
                self.trainer.save_checkpoint(checkpoint_path)
                self.finish_tasks.put(dict(id=task['id'], score=score))
            except KeyboardInterrupt:
                break


class Explorer(mp.Process):
    def __init__(self, mutation_count, mutation_search_max_count, population, finish_tasks):
        super().__init__()
        self.mutation_count = mutation_count
        self.population = population
        self.finish_tasks = finish_tasks
        self.mutation_search_max_count = mutation_search_max_count

    # Get k largest elements 
    def kLargest(self, arr, k, tasks):
        index_picked = randint(0,k-1)

        sorted_tops = sorted(tasks, key=lambda x: x['score'], reverse=True)
        return sorted_tops[index_picked]

    def run(self):
        # mentain a list of top performers 
        best_elements = []

        while True:
            if self.mutation_count.value > self.mutation_search_max_count:
                print("Reached mutation cout")
                break
            if self.population.empty() and self.finish_tasks.full():
                print("Exploit and explore")
                tasks = []
                
                while not self.finish_tasks.empty():
                    tasks.append(self.finish_tasks.get())
                tasks = sorted(tasks, key=lambda x: x['score'], reverse=True)
                print('Best score on', tasks[0]['id'], 'is', tasks[0]['score'])
                print('Worst score on', tasks[-1]['id'], 'is', tasks[-1]['score'])
                
                # Top 20% models mutated at random 
                fraction = 0.2
                cutoff = int(np.ceil(fraction * len(tasks)))
                
                tops = tasks[:cutoff]

                for task in tops:
                    torch.save(torch.load("checkpoints/task-%03d.pth" % task['id']), "checkpoints/task-%03d.pth" % (task['id']+9999))

                best_elements += tops

                bottoms = tasks[len(tasks) - cutoff:]

                for bottom in bottoms:
                    # Bottom make mutation of a random top state
                    top = self.kLargest(best_elements, len(tasks[:cutoff]), tasks)
                    top_checkpoint_path = "checkpoints/task-%03d.pth" % (top['id']+9999)
                    bot_checkpoint_path = "checkpoints/task-%03d.pth" % bottom['id']
                    exploit_and_explore(top_checkpoint_path, bot_checkpoint_path)
                    with self.mutation_count.get_lock():
                        self.mutation_count.value += 1
                for top in tops:
                    # Top just make mutation of their current state
                    top_checkpoint_path = "checkpoints/task-%03d.pth" % (top['id']+9999)
                    bot_checkpoint_path = "checkpoints/task-%03d.pth" % top['id']
                    exploit_and_explore(top_checkpoint_path, bot_checkpoint_path)
                    with self.mutation_count.get_lock():
                        self.mutation_count.value += 1
                for task in tasks:
                    # Add tasks to population for processing
                    self.population.put(task)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--device", type=str, default='cuda',
                        help="")
    parser.add_argument("--population_size", type=int, default=10,
                        help="")

    args = parser.parse_args()
    # mp.set_start_method("spawn")
    mp = mp.get_context('forkserver')
    device = 'cuda:'
    if not torch.cuda.is_available():
        device = 'cpu'

    population_size = args.population_size
    batch_size = 16
    mutation_search_max_count = 8

    pathlib.Path('checkpoints').mkdir(exist_ok=True)
    checkpoint_str = "checkpoints/task-%03d.pth"

    population = mp.Queue(maxsize=population_size)
    finish_tasks = mp.Queue(maxsize=population_size)

    mutation_count = mp.Value('i', 0)
    for i in range(population_size):
        population.put(dict(id=i, score=0))

    workers = []
    for i in range(0,2):
        workers.append(Worker(mutation_count, mutation_search_max_count, population, finish_tasks, f"cuda:{i}"))
  
    print("Created workers")
    workers.append(Explorer(mutation_count, mutation_search_max_count, population, finish_tasks))
    [w.start() for w in workers]
    print("Started all workers")
    [w.join() for w in workers]
    print("Finished all workers")
    task = []
    while not finish_tasks.empty():
        task.append(finish_tasks.get())
    while not population.empty():
        task.append(population.get())
    task = sorted(task, key=lambda x: x['score'], reverse=True)
    print('best score in last run: ', task[0]['id'], 'is', task[0]['score'])
