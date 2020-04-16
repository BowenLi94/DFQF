
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from sys import float_info
from collections import OrderedDict
from math import sqrt
import utils.q_utils as q_utils

from copy import deepcopy
import matplotlib.pyplot as plt
from .clip import *
import utils.quant_linear as ql

def has_children(module):
    try:
        next(module.children())
        return True
    except StopIteration:
        return False
def make_non_parallel_copy(model):
    """Make a non-data-parallel copy of the provided model.

    torch.nn.DataParallel instances are removed.
    """
    def replace_data_parallel(container):
        for name, module in container.named_children():
            if isinstance(module, nn.DataParallel):
                setattr(container, name, module.module)
            if has_children(module):
                replace_data_parallel(module)

    # Make a copy of the model, because we're going to change it
    new_model = deepcopy(model)
    if isinstance(new_model, nn.DataParallel):
        new_model = new_model.module
    replace_data_parallel(new_model)

    return new_model

class _QuantStatsRecord(object):
    @staticmethod
    def create_records_dict():
        records = OrderedDict()
        records['total_numel'] = 0
        records['clip_a'] = 0
        return records

    def __init__(self):
        # We don't know the number of inputs at this stage so we defer records creation to the actual callback
        self.inputs = []
        self.output = self.create_records_dict()

class LayerActivationsCollect:
    def __init__(self,model_layer,args):
        self.hook = model_layer.register_forward_hook(self.hook_fn)
        self.record = _QuantStatsRecord.create_records_dict()
        self.args = args
        
    def hook_fn(self, module, input, output):

        self.record['total_numel'] += 1
        act = input[0].data.cpu().numpy().flatten()
        if self.args.q_clip == 'max':
            self.record['clip_a'] += np.max(act)
        elif self.args.q_clip == 'mse':
            self.record['clip_a'] += find_clip_mmse(act,self.args.q_abit,True)
        elif self.args.q_clip == 'aciq':
            self.record['clip_a'] += find_clip_aciq(act,self.args.q_abit,True)
        elif self.args.q_clip == 'kl':
            self.record['clip_a'] += find_clip_entropy(act,self.args.q_abit,True)
        elif self.args.q_clip == 'ft':
            self.record['clip_a'] += max(np.max(act)*self.args.q_lb, find_clip_mmse(act,self.args.q_abit,True))
            #self.record['clip_a'] +=  np.max(act)

    def remove(self):
        self.hook.remove()

def run_forward(net,generator,cal_loader,args):

    net.eval()
    with torch.no_grad():
        if cal_loader is not None:
            for i, images in enumerate(cal_loader):
                if len(images) == 2:
                    images = images[0]
                images = images.cuda()
                output = net(images)
                if i == 0:
                    #print('break out!!!!!')
                    break
        else:
            for i in range(10):
                z = Variable(torch.randn(args.batch_size, args.latent_dim)).cuda()
                images = generator(z)
                output = net(images)


def collect_stats(net,generator,cal_loader,args):
    if not args.silent:
        if cal_loader is None:
            print('==> Collect statistics from Generator')
        else:
            print('==> Collect statistics from Dataloader')

    if torch.nn.DataParallel in [type(m) for m in net.modules()]:
        net = make_non_parallel_copy(net)

    namelist_relu = []
    modulelist_relu = []
    namelist_conv = []
    modulelist_conv = []
    for name, module in net.named_modules():
        if isinstance(module, nn.ReLU) :
            namelist_relu.append(name)
            modulelist_relu.append(module)
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            namelist_conv.append(name)
            modulelist_conv.append(module)

    hooklist = []
    for layer in modulelist_relu:
        hooklist.append(LayerActivationsCollect(layer,args))

    run_forward(net,generator,cal_loader,args)

    for hook in hooklist:
        hook.record['clip_a'] /= hook.record['total_numel']
        hook.remove()

    clip_dict = {}
    for i, key in enumerate(namelist_relu):
        clip_dict.update({key+'.clip_a': torch.Tensor([hooklist[i].record['clip_a']])})

    for i, key in enumerate(namelist_conv):
        value = modulelist_conv[i].weight.data.cpu().numpy().flatten()
        if isinstance( modulelist_conv[i], nn.Conv2d) and modulelist_conv[i].in_channels == 3:
            q_wbit = 8
        elif isinstance( modulelist_conv[i], nn.Linear) :
            q_wbit = 8 
        else:
            q_wbit = args.q_wbit

        if args.q_clip == 'max':
            clip_w = np.max(np.abs(value))
        elif args.q_clip == 'mse' :
            clip_w = find_clip_mmse(value,q_wbit,False)
        elif args.q_clip == 'aciq':
            clip_w = find_clip_aciq(value,q_wbit,False)
        elif args.q_clip == 'kl':
            clip_w = find_clip_entropy(value,q_wbit,False)
        elif args.q_clip == 'ft':
            #clip_w = np.max(np.abs(value))
            clip_w = max(np.max(np.abs(value))*args.q_lb, find_clip_mmse(value,q_wbit,False))
            #clip_w = find_clip_mmse(value,args.q_wbit,False)
        clip_dict.update({key+'.clip_w': torch.Tensor([clip_w])})

######################
    student = make_non_parallel_copy(net)

    def has_children(module):
        try:
            next(module.children())
            return True
        except StopIteration:
            return False
    def _pre_process_container(container, prefix=''):
        for name, module in container.named_children():
            full_name = prefix + name
            if isinstance(module, nn.ReLU):
                module_wrapper = ql.ReLU_DFQF(module, args.q_abit, clip_dict[full_name+'.clip_a'] )
                setattr(container, name, module_wrapper)
            elif isinstance(module, nn.Conv2d) :
                if module.in_channels == 3:
                    module_wrapper = ql.Conv2d_DFQF(module, 8, clip_dict[full_name+'.clip_w'])
                else:
                    module_wrapper = ql.Conv2d_DFQF(module, args.q_wbit, clip_dict[full_name+'.clip_w'])
                setattr(container, name, module_wrapper)
            elif isinstance(module, nn.Linear): 
                module_wrapper = ql.Linear_DFQF(module, 8, clip_dict[full_name+'.clip_w'])
                setattr(container, name, module_wrapper)
                
            if has_children(module):
                _pre_process_container(module, full_name + '.')
    

    _pre_process_container(student)

    return student
