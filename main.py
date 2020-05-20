### This is a modified version of main.py from https://github.com/facebookresearch/GradientEpisodicMemory
### The most significant changes are to the arguments: 1) allowing for new models and 2) changing the default settings in some cases

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import datetime
import argparse
import random
import copy
import uuid
import time
import os

import numpy as np

import torch
from torch.autograd import Variable
from metrics.metrics import confusion_matrix


# continuum iterator #########################################################


def load_datasets(args):
    d_tr, d_te = torch.load(args.data_path + '/' + args.data_file)
    n_inputs = d_tr[0][1].size(1)
    n_outputs = 0
    for i in range(len(d_tr)):
        n_outputs = max(n_outputs, d_tr[i][2].max())
        n_outputs = max(n_outputs, d_te[i][2].max())
    return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)


def add_list(list1, list2):
    for i in range(len(list1)):
        list1[i] += list2[i]

    return list1


class Continuum:
    # random samples from the buffer to be trained alongside the current example
    def __init__(self, data, args):
        self.data = data
        self.batch_size = args.batch_size
        n_tasks = len(data)
        task_permutation = range(n_tasks)

        if args.shuffle_tasks == 'yes':
            task_permutation = torch.randperm(n_tasks).tolist()

        sample_permutations = []

        # decide number of samples per task
        for t in range(n_tasks):
            N = data[t][1].size(0)
            if args.samples_per_task <= 0:
                n = N
            else:
                n = min(args.samples_per_task, N)

            p = torch.randperm(N)[0:n]
            sample_permutations.append(p)

        self.permutation = []

        # take the i-th sample in dataset of task_t
        for t in range(n_tasks):
            task_t = task_permutation[t]
            for _ in range(args.n_epochs):
                task_p = [[task_t, i] for i in sample_permutations[task_t]]
                random.shuffle(task_p)
                self.permutation += task_p

        self.length = len(self.permutation)
        self.current = 0

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.current >= self.length:
            raise StopIteration
        else:
            ti = self.permutation[self.current][0]
            j = []
            i = 0
            while (((self.current + i) < self.length) and
                   (self.permutation[self.current + i][0] == ti) and
                   (i < self.batch_size)):
                j.append(self.permutation[self.current + i][1])
                i += 1
            self.current += i
            j = torch.LongTensor(j)

            # return the next sample (j-th sample in task ti)
            return self.data[ti][1][j], ti, self.data[ti][2][j]


# train handle ###############################################################


def eval_tasks(model, tasks, args):
    current_dict = copy.deepcopy(model.state_dict())
    if os.path.exists(model.checkpoint_path):
        task_dict = torch.load(model.checkpoint_path)
    model.eval()
    result = []
    for i, task in enumerate(tasks):
        t = i
        x = task[1]
        y = task[2]
        rt = 0
        if 'expansion' in args.model \
                and os.path.exists(model.checkpoint_path) \
                and t < len(task_dict):
            # load checkpoint for evaluation
            model.load_state_dict(task_dict[str(t)])
        else:
            model.load_state_dict(current_dict)

        eval_bs = x.size(0)

        with torch.no_grad():  # torch 0.4+
            for b_from in range(0, x.size(0), eval_bs):
                b_to = min(b_from + eval_bs, x.size(0) - 1)
                if b_from == b_to:
                    xb = x[b_from].view(1, -1)
                    yb = torch.LongTensor([y[b_to]]).view(1, -1)
                else:
                    xb = x[b_from:b_to]
                    yb = y[b_from:b_to]
                if args.cuda:
                    xb = xb.cuda()
                # xb = Variable(xb, volatile=True)  # torch 0.4+
                _, pb = torch.max(model(xb, t).data.cpu(), 1, keepdim=False)
                rt += (pb == yb).float().sum()

        result.append(rt / x.size(0))

    model.load_state_dict(current_dict)

    return result


def life_experience(model, continuum, x_te, args):
    result_a = []  # accuracy
    result_t = []  # number of task
    result_l = []
    task_l = []
    cos_layer = []
    cos_weight = []

    current_task = 0
    batch_per_task = int(args.samples_per_task * args.n_epochs / args.batch_size)
    observe_batch = int(0.05 * batch_per_task)
    observe = 1
    temp_total_task = args.task_num

    time_start = time.time()
    train_start = time_start

    for (i, (x, t, y)) in enumerate(continuum):
        if t != current_task:
            if args.cuda:
                torch.cuda.empty_cache()
            print("Training done")
            print("Training time: ", time.time() - train_start)
            temp_acc = eval_tasks(model, x_te, args)[:temp_total_task+1]
            for pre_t in range(t):
                print("accuracy of task " + str(pre_t) + " is: " + str(temp_acc[pre_t].item()))
            if t > temp_total_task:  # hot fix
                break

            result_a.append(temp_acc)
            result_t.append(current_task)
            result_l.append(task_l)
            task_l = []
            cos_weight = []
            cos_layer = []
            current_task = t
            observe = 1

            print("\nTask " + str(t))
            print("Start observing...")

        if (i % args.log_every) == 0:
            result_a.append(eval_tasks(model, x_te, args)[:temp_total_task+1])
            result_t.append(current_task)

        if 'expansion' in args.model \
                and (i == batch_per_task * t + observe_batch) and t > 0:
            print("Observation done")
            print("Start expanding...")
            cos_layer = torch.tensor(cos_layer)
            if args.cuda:
                cos_layer = cos_layer.cuda()
            model.expand(cos_layer, cos_weight, t)
            print("Expanding done")
            train_start = time.time()
            print("Start training...")
            observe = 0
            if args.cuda:
                try:
                    model.cuda()
                except:
                    pass

        v_x = x.view(x.size(0), -1)
        v_y = y.long()

        if args.cuda:
            v_x = v_x.cuda()
            v_y = v_y.cuda()

        if 'expansion' in args.model and observe and t > 0:
            model.train()
            temp_layer, temp_weight = model.observe(Variable(v_x), t, Variable(v_y))
            cos_layer.append(temp_layer)
            if cos_weight == []:
                cos_weight = temp_weight
            else:
                cos_weight = add_list(cos_weight, temp_weight)
        else:
            model.train()
            task_l.append(model.update(Variable(v_x), t, Variable(v_y)))

    result_a.append(eval_tasks(model, x_te, args)[:temp_total_task+1])
    result_t.append(current_task)
    if 'expansion' in args.model:
        result_l[0] = result_l[0][observe_batch:] # hot fix

    time_end = time.time()
    time_spent = time_end - time_start

    return torch.Tensor(result_t), torch.Tensor(result_a), torch.Tensor(result_l), time_spent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continuum learning')

    # model details
    parser.add_argument('--model', type=str, default='expansion_v2',
                        help='model to train')
    parser.add_argument('--n_hiddens', type=int, default=100,
                        help='number of hidden neurons at each layer')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--finetune', default='yes', type=str, help='whether to initialize nets in indep. nets')

    # optimizer parameters influencing all models
    parser.add_argument('--n_epochs', type=int, default=2,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='the amount of items received by the algorithm at one time (set to 1 across all experiments). Variable name is from GEM project.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')

    # memory parameters for GEM baselines
    parser.add_argument('--n_memories', type=int, default=25,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', default=1.0, type=float,
                        help='memory strength (meaning depends on memory)')

    # expansion model parameters
    parser.add_argument('--thre', type=float, default=1,
                        help='Threshold to decide expand or not')
    parser.add_argument('--expand_size', type=float, nargs='+', default=[0.4, 0.2],
                        help='Percent of neurons to expand')
    parser.add_argument('--freeze_all', type=str, default='yes',
                        help='Freeze all neurons from previous task or not')
    parser.add_argument('--task_num', type=int, default=6,
                        help='Number of tasks to observe')
    parser.add_argument('--mode', type=str, default="sort",
                        help='Method to select neurons. sort & random & mask')

    # experiment parameters
    parser.add_argument('--cuda', type=str, default='no',
                        help='Use GPU?')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed of model')
    parser.add_argument('--log_every', type=int, default=100,
                        help='frequency of logs, in minibatches')
    parser.add_argument('--save_path', type=str, default='results/',
                        help='save models at the end of training')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/',
                        help='save models of each task')

    # data parameters
    parser.add_argument('--data_path', default='data/',
                        help='path where data is located')
    parser.add_argument('--data_file', default='cifar100_20_o.pt',
                        help='data file')
    parser.add_argument('--samples_per_task', type=int, default=50,
                        help='training samples per task (all if negative)')
    parser.add_argument('--shuffle_tasks', type=str, default='no',
                        help='present tasks in order')
    args = parser.parse_args()

    args.cuda = True if args.cuda == 'yes' else False
    args.finetune = True if args.finetune == 'yes' else False
    args.freeze_all = True if args.freeze_all == 'yes' else False

    # taskinput model has one extra layer
    if args.model == 'taskinput':
        args.n_layers -= 1

    # unique identifier
    # uid = uuid.uuid4().hex

    # initialize seeds
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        print("Found GPU:", torch.cuda.get_device_name(0))
        torch.cuda.manual_seed_all(args.seed)

    # load data
    print("Loading Data")
    x_tr, x_te, n_inputs, n_outputs, n_tasks = load_datasets(args)
    n_outputs = n_outputs.item()  # outputs should not be a tensor, otherwise "TypeError: expected Float (got Long)"
    print("Data has been loaded")

    # set up continuum
    continuum = Continuum(x_tr, args)

    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    if args.cuda:
        try:
            model.cuda()
        except:
            pass

            # run model on continuum
    result_t, result_a, result_l, spent_time = life_experience(
        model, continuum, x_te, args)

    # prepare saving path and file name
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    fname = args.model + '_' + args.data_file + '_'
    fname += datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # fname += '_' + uid
    fname = os.path.join(args.save_path, fname)

    # save confusion matrix and print one line of stats
    stats = confusion_matrix(result_t, result_a, fname + '.txt')
    one_liner = str(vars(args)) + ' # '
    one_liner += ' '.join(["%.3f" % stat for stat in stats])
    print(fname + ': ' + one_liner + ' # ' + str(spent_time))

    # save all results in binary file
    torch.save((result_t, result_a, result_l, stats, one_liner, args), fname + '.pt')
