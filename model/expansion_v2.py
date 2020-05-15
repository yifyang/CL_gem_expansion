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
import copy

from .common import MLP, vgg11_bn


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


def store_layer_grad(layers, grads_layer, grad_dims_layer, tid):
    """
        This stores parameter gradients at each layers of past tasks.
        layers: layers in neural network
        grads_layer: gradients at each layer
        grad_dims_layer: list with number of parameters per layers
        tid: task id
    """
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


def layer_sort(cos_layer, t, threshold, ass):
    """
        This sort the gradient of layers.
        cos_layer: cosine similarity between two tasks at each layer
        t: index of current training task
    """
    layers = len(cos_layer[0])
    layers_cos = [0] * layers

    for i in range(layers):
        temp = torch.sum(torch.sum(cos_layer[:, i], dim=0) / len(cos_layer))
        layers_cos[i] = temp / t

    layers_sort_cos, layers_sort = torch.sort(torch.tensor(layers_cos))

    layers_expand = [0] * layers
    j = 0
    for i in range(layers):
        if layers_sort[i] == 0:
            continue
        elif layers_cos[layers_sort[i]] > threshold:
            layers_expand[layers_sort[i]] = 0
            j += 1
            continue
        else:
            if type(ass) is not list: # assign the same expanding rate to layers
                layers_expand[layers_sort[i]] = ass
            else:
                layers_expand[layers_sort[i]] = ass[j]
            j += 1

        if type(ass) is list:
            print("layer to expand: " + str(layers_sort[i]) + " ; "
                  + str(layers_expand[layers_sort[i]]))
            print("cos distance: " + str(layers_sort_cos[i]))

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
            self.net = vgg11_bn()
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        self.n_hiddens = []
        for param in self.parameters():
            if len(param.size()) > 1:
                self.n_hiddens.append(param.size()[1])

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.n_tasks = n_tasks
        self.gpu = args.cuda
        self.lr = args.lr
        self.thre = args.thre
        if self.is_cifar:
            self.expand_size = args.expand_size[0]
        else:
            self.expand_size = args.expand_size

        self.task_dict = []  # store dict for each task
        self.neuron_share = []

        # allocate episodic memory
        self.memory_data = torch.FloatTensor(
            self.n_tasks, self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(self.n_tasks, self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        self.allocate()
        self.sel_neuron = [[]]
        self.frz_neuron = [[np.array([])] * (len(self.for_layer) + 1)]
        for param in self.for_layer:
            param_size = param.size()
            self.sel_neuron[0].append(np.arange(param_size[1]))
        self.sel_neuron[0] += [np.arange(n_outputs)]

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

        self.for_layer = []
        self.grad_dims_layer = []
        layer_num = 0
        self.grads_layer = []
        for name, param in self.named_parameters():
            if 'bias' not in name and len(param.size()) > 1:
                self.for_layer.append(param)
        for param in self.for_layer:
            self.grad_dims_layer.append([param.data.numel()])
            self.grads_layer.append(torch.Tensor(sum(self.grad_dims_layer[layer_num]), self.n_tasks))
            if self.gpu:
                self.grads_layer[-1] = self.grads_layer[-1].cuda()
            layer_num += 1

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
        layers_expand = layer_sort(cos_layer, t, self.thre, self.expand_size)
        layer_size = []
        new_dict = copy.deepcopy(self.state_dict())

        layer = 0
        weight_sort = []
        for name, param in self.named_parameters():
            # expand the number of neurons at each layer
            param_size = param.size()
            if 'bias' not in name and len(param_size) > 1:
                # sort gradient of weights at each layer
                if len(param_size) > 2:
                    temp_weight = torch.sum(cos_weight[layer].view(param_size), dim=0)
                    temp_weight = torch.sum(temp_weight, dim=1)
                    cos_weight[layer] = torch.sum(temp_weight, dim=1)
                else:
                    cos_weight[layer] = torch.sum(cos_weight[layer].view(param_size), dim=0)
                _, temp_sort = torch.sort(cos_weight[layer])

                # select neurons used in the last task
                sel_sort = []
                for index in temp_sort:
                    if index.item() in self.sel_neuron[t-1][layer]:
                        sel_sort.append(index.item())
                weight_sort.append(np.array(sel_sort))
                layer += 1

        # Both of testing and training based on continuously expanded network
        j = -1
        hidden_layer_linear = []
        hidden_layer_conv = []
        share_neuron = []
        freeze_neuron = []
        pre_x = self.n_inputs
        # for name, param in self.named_parameters():
        for name in new_dict:
            param = new_dict[name]
            # expand the number of neurons at each layer
            param_size = param.size()
            if 'num_batches_tracked' in name:
                continue
            elif 'bias' not in name and len(param_size) > 1:
                j += 1
                layer_size.append(param_size)
                if j == 0:
                    # expand the first layer
                    copy_neuron_x = np.array(weight_sort[j+1][:int(layers_expand[j+1] * self.n_hiddens[j+1])])

                    # share all neurons at the first layer
                    share_neuron.append(np.arange(self.n_inputs))
                    freeze_neuron.append((np.array([])))

                    init_weight = param[copy_neuron_x, :]
                    if self.gpu:
                        init_weight = init_weight.cuda()
                    new_dict[name] = torch.cat((new_dict[name], init_weight), 0)
                    new_dict[name][copy_neuron_x, :] = 0
                elif j == layer-1:
                    # expand the last layer
                    copy_neuron_y = np.array(weight_sort[j][:int(layers_expand[j] * self.n_hiddens[j])])
                    temp_share_neuron = np.append(
                        weight_sort[j][int(layers_expand[j] * self.n_hiddens[j]):],
                        np.arange(param_size[0], pre_x))
                    temp_freeze_neuron = np.arange(param_size[0])
                    share_neuron.append(np.array(temp_share_neuron))
                    freeze_neuron.append(temp_freeze_neuron)

                    # share all neurons at the last layer
                    share_neuron.append(np.arange(self.n_outputs))
                    freeze_neuron.append((np.array([])))

                    # copy neurons from old ones
                    init_weight = param[:, copy_neuron_y]
                    if self.gpu:
                        init_weight = init_weight.cuda()
                    new_dict[name] = torch.cat((new_dict[name], init_weight), 1)
                    new_dict[name][:, copy_neuron_y] = 0

                    # assign numbers of neurons in expanded network
                    if self.is_cifar and 'features' in name:
                        hidden_layer_conv.append(expand_y)
                    elif self.is_cifar and 'classifier' in name:
                        hidden_layer_linear.append(expand_y)
                    else:
                        hidden_layer_linear.append(new_dict[name].shape[1])
                else:
                    # expand the hidden layers
                    expand_x, expand_y = int(layers_expand[j+1] * self.n_hiddens[j+1] + param_size[0]), pre_x

                    if self.is_cifar and 'features' in name:
                        hidden_layer_conv.append(expand_y)
                    elif self.is_cifar and 'classifier' in name:
                        hidden_layer_linear.append(expand_y)
                    else:
                        hidden_layer_linear.append(expand_y)

                    # select neurons to be activated and frozen for the coming task
                    copy_neuron_y = np.array(weight_sort[j][:int(layers_expand[j] * self.n_hiddens[j])])
                    copy_neuron_x = np.array(weight_sort[j+1][:int(layers_expand[j+1] * self.n_hiddens[j+1])])
                    temp_share_neuron = np.append(
                        weight_sort[j][int(layers_expand[j] * self.n_hiddens[j]):],
                        np.arange(param_size[1], expand_y))
                    temp_freeze_neuron = np.arange(param_size[1])
                    share_neuron.append(temp_share_neuron)
                    freeze_neuron.append(temp_freeze_neuron)

                    # expand the layer
                    expand_size = list(param_size)
                    expand_size[0] = expand_x
                    expand_size[1] = expand_y
                    init_weight = torch.zeros(expand_size)
                    if self.gpu:
                        init_weight = init_weight.cuda()
                    torch.nn.init.xavier_normal_(init_weight, gain=1)
                    # copy neurons from old ones
                    init_weight[:param_size[0], :param_size[1]] = param
                    init_weight[param_size[0]:, :param_size[1]] = param[copy_neuron_x, :]
                    init_weight[:param_size[0], param_size[1]:] = param[:, copy_neuron_y]
                    init_weight[copy_neuron_x, :] = 0
                    init_weight[:, copy_neuron_y] = 0
                    new_dict[name] = init_weight
                pre_x = new_dict[name].shape[0]
            else:
                if j == layer-1:
                    j += 1
                    continue
                else:
                    init_bias = param[copy_neuron_x]
                    if self.gpu:
                        init_bias = init_bias.cuda()
                    new_dict[name] = torch.cat((new_dict[name], init_bias), 0)
                    new_dict[name][copy_neuron_x] = 0

        self.sel_neuron.append(share_neuron)
        self.frz_neuron.append(freeze_neuron)

        self.share(new_dict, t)

        # rebuild the network
        if self.is_cifar:
            self.net = vgg11_bn(hidden_layer_linear,
                                hidden_layer_conv+[hidden_layer_conv[-1]])
        else:
            self.net = MLP([self.n_inputs] + hidden_layer_linear + [self.n_outputs])
        self.load_state_dict(new_dict)
        self.opt = optim.SGD(self.parameters(), self.lr)
        self.allocate()

        if self.gpu:
            self.cuda()


    def share(self, new_dict, t):
        current_dict = copy.deepcopy(self.state_dict())
        self.task_dict.append(current_dict)

        # update the expanded neurons in previous state_dict(set to be 0)
        # to fit in the expanded network
        for t_i in range(t):
            for name in self.task_dict[t_i]:
                param_size = new_dict[name].size()
                pre_size = self.task_dict[t_i][name].size()
                if 'num_batches_tracked' in name:
                    continue
                elif 'bias' not in name and len(param_size) > 1:
                    if len(param_size) > 2:
                        cat_weight = torch.zeros(param_size[0]-pre_size[0], pre_size[1],
                                                 pre_size[2], pre_size[3])
                    else:
                        cat_weight = torch.zeros(param_size[0]-pre_size[0], pre_size[1])
                    if self.gpu:
                        cat_weight = cat_weight.cuda()
                    self.task_dict[t_i][name] = torch.cat((self.task_dict[t_i][name], cat_weight), 0)

                    if len(param_size) > 2:
                        cat_weight = torch.zeros(param_size[0], param_size[1]-pre_size[1],
                                                 pre_size[2], pre_size[3])
                    else:
                        cat_weight = torch.zeros(param_size[0], param_size[1]-pre_size[1])
                    if self.gpu:
                        cat_weight = cat_weight.cuda()
                    self.task_dict[t_i][name] = torch.cat((self.task_dict[t_i][name], cat_weight), 1)
                else:
                    cat_bias = torch.zeros(param_size[0]-pre_size[0])
                    if self.gpu:
                        cat_bias = cat_bias.cuda()
                    self.task_dict[t_i][name] = torch.cat((self.task_dict[t_i][name], cat_bias), 0)

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

        # freeze neurons
        if t > 0:
            layer = -1
            for name, param in self.named_parameters():
                if 'bias' not in name and len(param.size()) > 1:
                    layer += 1
                    param.grad[:, self.frz_neuron[t][layer]] = 0
                    param.grad[self.frz_neuron[t][layer+1], :] = 0
                else:
                    param.grad[self.frz_neuron[t][layer+1]] = 0

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
                store_layer_grad(self.for_layer, self.grads_layer,
                                 self.grad_dims_layer, past_task)

        # now compute the grad on the current minibatch
        self.zero_grad()

        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
        loss.backward()

        # check if gradient violates constraints
        cos_layers = [[0] * (self.n_tasks - 1)] * len(self.grads_layer)  # record cos and return
        cos_weight = []
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_layer_grad(self.for_layer, self.grads_layer,
                             self.grad_dims_layer, t)

            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                else torch.LongTensor(self.observed_tasks[:-1])

            cos_layers = []
            cos_weight = []
            for layer_num in range(len(self.grads_layer)):
                num_weights = len(self.grads_layer[layer_num][:, t])
                cos_layer_temp = []
                cos_weight_temp = torch.zeros(num_weights)
                if self.gpu:
                    cos_weight_temp = cos_weight_temp.cuda()
                for pre_task in indx:
                    cur_grad = self.grads_layer[layer_num][:, t]
                    pre_grad = self.grads_layer[layer_num][:, pre_task]
                    dotp_weight = torch.mul(cur_grad, pre_grad)
                    pre_weight_norm = torch.mul(pre_grad, pre_grad)
                    cur_weight_norm = torch.mul(cur_grad, cur_grad)
                    # compute the cosine similarity at layer level
                    cos_layer_temp.append((torch.sum(dotp_weight)
                                          / (torch.sum(cur_weight_norm).sqrt()
                                             * torch.sum(pre_weight_norm).sqrt())).item())

                    # compute the cosine similarity at weight level
                    weight_norm = torch.mul(cur_weight_norm.sqrt(), pre_weight_norm.sqrt())
                    task_weight_temp = torch.div(dotp_weight, weight_norm)
                    cos_weight_temp += task_weight_temp
                cos_layer_temp += [0] * ((self.n_tasks - 1) - len(cos_layer_temp))
                cos_layers.append(cos_layer_temp)
                cos_weight.append(cos_weight_temp)

        self.zero_grad()

        return cos_layers, cos_weight
