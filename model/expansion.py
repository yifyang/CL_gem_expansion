### This is a copy of GEM from https://github.com/facebookresearch/GradientEpisodicMemory.
### In order to ensure complete reproducability, we do not change the file and treat it as a baseline.

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import quadprog

from .common import MLP, ResNet18


# Auxiliary functions useful for GEM's inner optimization.

def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def store_layer_grad(layers, grads_layer, grad_dims_layer, tid, is_cifar):
    """
        This stores parameter gradients at each layers of past tasks.
        layers: layers in neural network
        grads_layer: gradients at each layer
        grad_dims_layer: list with number of parameters per layers
        tid: task id
    """
    if is_cifar:
        layer_num = 0
        for layer in layers:
            grads_layer[layer_num][:, tid].fill_(0.0)
            cnt = 0
            for param in layer.parameters():
                if param.grad is not None:
                    beg = 0 if cnt == 0 else sum(grad_dims_layer[layer_num][:cnt])
                    en = sum(grad_dims_layer[layer_num][:cnt + 1])
                    grads_layer[layer_num][beg: en, tid].copy_(param.grad.data.view(-1))
                cnt += 1
            layer_num += 1
    else:
        layer_num = 0
        for param in layers:
            grads_layer[layer_num][:, tid].fill_(0.0)
            cnt = 0
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims_layer[layer_num][:cnt])
                en = sum(grad_dims_layer[layer_num][:cnt + 1])
                grads_layer[layer_num][beg: en, tid].copy_(param.grad.data.view(-1))
            cnt += 1
            layer_num += 1

def layer_sort(cos_layer, t):
    """
        This sort the gradient of layers.
        cos_layer: cosine similarity between two tasks at each layer
        t: index of current training task
    """
    layers = len(cos_layer[0])
    layers_cos = [0] * layers
    ass = [0.5, 0.2]

    for i in range(layers):
        temp = torch.sum(torch.sum(cos_layer[:, i], dim=0) / len(cos_layer))
        layers_cos[i] = temp / t

    layers_sort_cos, layers_sort = torch.sort(torch.tensor(layers_cos))

    layers_expand = [0] * (layers - 1)
    j = 0
    for i in range(layers):
        if layers_sort[i] == 0:
            continue
        elif layers_cos[layers_sort[i]] > 0.01:
            layers_expand[layers_sort[i] - 1] = 0
            j += 1
            print("layer to expand: " + str(layers_sort[i]) + " ; " + str(0))
            print("cos distance: " + str(layers_sort_cos[i]))
            continue
        else:
            layers_expand[layers_sort[i] - 1] = ass[j]
            print("layer to expand: " + str(layers_sort[i]) + " ; " + str(ass[j]))
            print("cos distance: " + str(layers_sort_cos[i]))
            j += 1

    return layers_expand


class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.margin = args.memory_strength
        self.is_cifar = (args.data_file == 'cifar100.pt' or
                         args.data_file == 'cifar100_20.pt' or
                         args.data_file == 'cifar100_20_o.pt')
        if self.is_cifar:
            self.net = ResNet18(n_outputs)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        self.sel_neuron = [[]] * nl
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        self.n_layers = nl

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.n_tasks = n_tasks
        self.gpu = args.cuda
        self.lr = args.lr

        # allocate episodic memory
        self.memory_data = torch.FloatTensor(
            self.n_tasks, self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(self.n_tasks, self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        self.allocate()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs

        if args.cuda:
            self.cuda()

    def allocate(self):
        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), self.n_tasks)
        if self.is_cifar:
            layers = [self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]
            self.for_layer = layers
            self.grad_dims_layer = []
            layer_num = 0
            self.grads_layer = []
            for layer in layers:
                self.grad_dims_layer.append([])
                for param in layer.parameters():
                    self.grad_dims_layer[layer_num].append(param.data.numel())
                self.grads_layer.append(torch.Tensor(sum(self.grad_dims_layer[layer_num]), self.n_tasks))
                if self.gpu:
                    self.grads_layer[-1] = self.grads_layer[-1].cuda()
                layer_num += 1
        else:
            self.for_layer = []
            self.grad_dims_layer = []
            layer_num = 0
            self.grads_layer = []
            for name, param in self.named_parameters():
                if 'bias' not in name:
                    self.for_layer.append(param)
            for param in self.for_layer:
                self.grad_dims_layer.append([param.data.numel()])
                self.grads_layer.append(torch.Tensor(sum(self.grad_dims_layer[layer_num]), self.n_tasks))
                if self.gpu:
                    self.grads_layer[-1] = self.grads_layer[-1].cuda()
                layer_num += 1

        if self.gpu:
            self.grads = self.grads.cuda()

    def forward(self, x, t):
        output = self.net(x)
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def expand(self, cos_layer, cos_weight, t):
        layers = len(cos_layer[0])
        layers_expand = layer_sort(cos_layer, t)
        layer_size = []
        new_dict = self.state_dict()
        self.sel_neuron = [[]] * self.n_layers

        j = 0
        weight_sort = []
        hidden_layer = []
        pre_x = self.n_inputs
        for name, param in self.named_parameters():
            # expand the number of neurons at each layer
            temp_size = param.size()
            if 'bias' not in name:
                expand_x, expand_y = temp_size[0], pre_x
                layer_size.append(temp_size)
                if j < len(layers_expand):
                    # expand weights
                    expand_x = int((layers_expand[j]+1)*temp_size[0])
                    hidden_layer.append(expand_x)
                    init_weight = torch.zeros(expand_x, expand_y)
                    if self.gpu:
                        init_weight = init_weight.cuda()
                    torch.nn.init.xavier_normal_(init_weight, gain=1)
                    init_weight[:temp_size[0], :temp_size[1]] = new_dict[name]
                    new_dict[name] = nn.Parameter(init_weight)

                    # expand bias
                    init_bias = torch.zeros(expand_x)
                    if self.gpu:
                        init_bias = init_bias.cuda()
                    init_bias[:temp_size[0]] = new_dict[name.replace('weight', 'bias')]
                    new_dict[name.replace('weight', 'bias')] = \
                        nn.Parameter(init_bias)
                    pre_x = expand_x
                elif j >= len(layers_expand):
                    init_weight = torch.zeros(expand_x, expand_y)
                    if self.gpu:
                        init_weight = init_weight.cuda()
                    torch.nn.init.xavier_normal_(init_weight, gain=1)
                    init_weight[:temp_size[0], :temp_size[1]] = new_dict[name]
                    new_dict[name] = nn.Parameter(init_weight)

                # sort gradient of weights at each layer
                cos_weight[j] = torch.sum(cos_weight[j].view(temp_size[0], temp_size[1]), dim=1)
                if j < len(layers_expand):
                    _, temp_sort = torch.sort(cos_weight[j])
                    weight_sort.append(temp_sort)
                j += 1

        # select neurons
        for ii in range(layers-1):
            self.sel_neuron[ii] = [False] * int(len(cos_weight[ii])
                                             * (1 + layers_expand[ii]))
            weight_sort[ii] = weight_sort[ii][:int(layers_expand[ii] * len(cos_weight[ii]))]
            for jj in range(len(self.sel_neuron[ii])):
                if jj in weight_sort[ii]:
                    self.sel_neuron[ii][jj] = True
        self.sel_neuron = [[False] * self.n_inputs] + self.sel_neuron + [[False] * self.n_outputs]

        # rebuild the network
        self.net = MLP([self.n_inputs] + hidden_layer + [self.n_outputs])
        self.load_state_dict(new_dict)
        self.opt = optim.SGD(self.parameters(), self.lr)
        self.allocate()

        if self.gpu:
            self.cuda()

    def update(self, x, t, y):
        # update memory
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[t, self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # now compute the grad on the current minibatch
        self.zero_grad()

        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
        loss.backward()

        # freeze the neurons selected
        if t > 0:
            layer = 0
            for name, param in self.named_parameters():
                mask1 = torch.Tensor(self.sel_neuron[layer]).long().nonzero().view(-1).numpy()
                mask2 = torch.Tensor(self.sel_neuron[layer+1]).long().nonzero().view(-1).numpy()
                if 'bias' not in name:
                    param.grad[:, mask1] = 0
                    param.grad[mask2, :] = 0
                    layer += 1
                else:
                    layer += 1
                    if layer >= self.n_layers:
                        break
                    param.grad[mask2] = 0


        self.opt.step()

        return loss

    def observe(self, x, t, y):
        # update memory
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[t, self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]

                offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                                   self.is_cifar)
                ptloss = self.ce(
                    self.forward(
                        Variable(self.memory_data[past_task]),
                        past_task)[:, offset1: offset2],
                    Variable(self.memory_labs[past_task] - offset1))
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                           past_task)
                store_layer_grad(self.for_layer, self.grads_layer,
                                 self.grad_dims_layer, past_task, self.is_cifar)

        # now compute the grad on the current minibatch
        self.zero_grad()

        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
        loss.backward()

        # check if gradient violates constraints
        cos_layers = [[0] * (self.n_tasks - 1)] * len(self.grads_layer) # record cos and return
        cos_weight = []
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            store_layer_grad(self.for_layer, self.grads_layer,
                             self.grad_dims_layer, t, self.is_cifar)

            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                else torch.LongTensor(self.observed_tasks[:-1])

            cos_layers = []
            cos_weight = []
            for layer_num in range(len(self.grads_layer)):
                num_weights = len(self.grads_layer[layer_num][:, t])
                cos_layer_temp = []
                cos_weight_temp = torch.zeros(num_weights)
                for pre_task in indx:
                    # compute the cosine similarity at layer level
                    cos_layer_temp.append(
                        torch.cosine_similarity(self.grads_layer[layer_num][:, t],
                                                self.grads_layer[layer_num][:, pre_task],
                                                dim=0).item())

                    # compute the cosine similarity at weight level
                    task_weight_temp = []
                    for i in range(num_weights):
                        task_weight_temp.append(
                            torch.cosine_similarity(self.grads_layer[layer_num][:, t][i],
                                                    self.grads_layer[layer_num][:, pre_task][i],
                                                    dim=0).item())
                    cos_weight_temp += torch.tensor(task_weight_temp)
                cos_layer_temp += [0] * ((self.n_tasks - 1) - len(cos_layer_temp))
                cos_layers.append(cos_layer_temp)
                cos_weight.append(cos_weight_temp)

        self.zero_grad()

        return cos_layers, cos_weight
