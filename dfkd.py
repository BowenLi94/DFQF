import os
import sys
import argparse
import numpy as np
import random
import math

import logging
from tensorboardX import SummaryWriter
import time


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import models
from utils import dataset
from utils import misc 


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', choices=['MNIST','cifar10','cifar100','imagenet'])
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.005, help='learning rate')
parser.add_argument('--lr_S', type=float, default=0.1, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=1000, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')

parser.add_argument('--output_dir', type=str, default='./checkpoint/')
parser.add_argument('--gpus', default='0')

parser.add_argument('--t_model', type=str, default='resnet20')
parser.add_argument('--s_model', type=str, default='resnet20')
parser.add_argument('--p_is', type=float, default=40., help='activation loss')
parser.add_argument('--p_adv', type=float, default=5., help='activation loss')
parser.add_argument('--p_bn', type=float, default=0.1, help='activation loss')

parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--logger', action='store_true',default=False, help='')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

img_shape = (args.channels, args.img_size, args.img_size)

cuda = True 
try:
    args.gpus = [int(s) for s in args.gpus.split(',')]
except ValueError:
    raise ValueError('ERROR: Argument --gpus must be a comma-separated list of integers only')
available_gpus = torch.cuda.device_count()
for dev_id in args.gpus:
    if dev_id >= available_gpus:
        raise ValueError('ERROR: GPU device ID {0} requested, but only {1} devices available'
                            .format(dev_id, available_gpus))
# Set default device in case the first one on the list != 0
torch.cuda.set_device(args.gpus[0])


if args.logger:
    timestr = time.strftime("%Y.%m.%d-%H%M%S")
    os.mkdir("./logs/"+timestr)
    args.output_dir = "./logs/"+timestr
    logging.basicConfig(filename="./logs/"+timestr+'/log', filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
    logging.info(args)

accr = 0
accr_best = 0

print('==>load data')
_, data_test_loader = dataset.get_dataset(args)

generator = models.generator.Generator(args).cuda()
generator = nn.DataParallel(generator,device_ids=args.gpus)

teacher = models.get_model(args.t_model,args).cuda()
teacher.load_state_dict(torch.load('./checkpoint/' + args.t_model + '_' + args.dataset + '.pth'))
teacher.eval()
teacher = nn.DataParallel(teacher,device_ids=args.gpus)

criterion = torch.nn.CrossEntropyLoss().cuda()


net = models.get_model(args.s_model,args).cuda()
net = nn.DataParallel(net,device_ids=args.gpus)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G)
optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_S, [0.4*args.n_epochs,0.8*args.n_epochs], gamma=0.1)


fdr_list = []
def hook_fn(module,input,output):
    input_per_channel = input[0].permute(1,0,2,3).flatten(start_dim=1)
    mean = input_per_channel.mean(dim =1)
    var  = input_per_channel.var(dim = 1)
    var[var==0] = 1e-8

    bn_kl = (var - module.running_var * torch.log(var)) + torch.pow((mean-module.running_mean),2)
    fdr_list.append(bn_kl.mean())

 
def get_hook_BN(net):
    for name, module in net.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.register_forward_hook(hook_fn)

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)   / y.shape[0]
    return l_kl

def test(net,testloader):
    net.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = correct/total
    loss = test_loss/(batch_idx+1)
    return acc, loss

print('==>Test teacher')
T_acc,T_loss = test(teacher, data_test_loader)
print('Teacher Test. Loss: %f, Accuracy: %f' % (T_loss, T_acc))
# ----------
#  Training
# ----------
if args.logger:
    tb_writer = SummaryWriter(log_dir='./logs/'+timestr)

if args.p_bn != 0:
    get_hook_BN(teacher)

for epoch in range(args.n_epochs):
    lr_scheduler.step(epoch)

    for i in range(120):
        loss_inception_score = torch.Tensor([0]).cuda()
        loss_bn_statistics = torch.Tensor([0]).cuda()
        loss_adversarial = torch.Tensor([0]).cuda()
        net.train()
        if args.p_is == 0 and args.p_bn == 0 and args.p_adv == 0:
            with torch.no_grad():
                z = Variable(torch.randn(args.batch_size, args.channels, args.img_size, args.img_size)).cuda()
                gen_imgs = torch.nn.BatchNorm2d(args.channels).cuda()(z)
                outputs_T = teacher(gen_imgs, out_feature=True)   

        else:
            z = Variable(torch.randn(args.batch_size, args.latent_dim)).cuda()
            optimizer_G.zero_grad()
            gen_imgs = generator(z)
            outputs_T = teacher(gen_imgs)

            loss = 0
            if args.p_is != 0:
                pyx = F.softmax(outputs_T, dim = 1)
                py  = pyx.mean(dim=0)
                loss_inception_score = -torch.nn.KLDivLoss(reduction='sum')(args.p_is *torch.log(py), pyx) / outputs_T.shape[0]
                loss += loss_inception_score

            if args.p_bn != 0:
                loss_bn_statistics = 0
                for fdr in fdr_list :
                    loss_bn_statistics += fdr.cuda()
                fdr_list = []
                loss +=  args.p_bn * loss_bn_statistics

            if args.p_adv != 0.:
                loss_adversarial = kdloss(net(gen_imgs), outputs_T) 
                loss -= args.p_adv * loss_adversarial
            loss.backward()
            optimizer_G.step()


        optimizer_S.zero_grad()        
        loss_kd = kdloss(net(gen_imgs.detach()), outputs_T.detach()) 
        loss_kd.backward()
        optimizer_S.step() 

        if i  == 0:
            print ("[Epoch %d/%d] [loss_is: %f] [loss_bn: %f] [loss_adv: %f] [loss_kd: %f]" % (epoch, args.n_epochs,loss_inception_score.item(), loss_bn_statistics.item(), loss_adversarial.item(), loss_kd.item()))
            if args.logger:
                logging.info("[Epoch %d/%d] [loss_is: %f] [loss_bn: %f] [loss_adv: %f] [loss_kd: %f]" % (epoch, args.n_epochs,loss_inception_score.item(), loss_bn_statistics.item(), loss_adversarial.item(), loss_kd.item()))
            
    S_acc, S_loss = test(net, data_test_loader)
    print('Test Avg. Loss: %f, Accuracy: %f' % (S_loss, S_acc))
    if args.logger:
        logging.info('Test Avg. Loss: %f, Accuracy: %f' % (S_loss, S_acc))
        tb_writer.add_scalar('scalar/test_accuracy',S_acc,epoch)
    if S_acc > accr_best:
        torch.save(net.state_dict(),args.output_dir + '/student_'+args.t_model+'_'+args.dataset+'.pth')
        torch.save(generator.state_dict(),args.output_dir + '/generator_'+args.t_model+'_'+args.dataset+'.pth')
        accr_best = S_acc

if args.logger:
    tb_writer.close()         
