import os

import time
import argparse
import logging

from utils import dataset
from utils import misc 
from utils import collect 
import models

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader


import logging
import numpy as np

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
# Common
parser.add_argument('--dataset', default='cifar10', choices=['cifar10','cifar100'])
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 64)')
parser.add_argument('--logger', action='store_true',default=False, help='')
parser.add_argument('--silent', action='store_true',default=False, help='')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Teacher
parser.add_argument('--t_model', default='resnet20', help='Teacher model')
parser.add_argument('--t_dir', default='', help='Teacher model')

# Student
parser.add_argument('--s_model', default='resnet18', help='Student model')
parser.add_argument("--s_lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument('--s_epochs', type=int, default=40, help='number of epochs to train (default: 10)')

# Generator
parser.add_argument("--latent_dim", type=int, default=1000, help="dimensionality of the latent space")

#mode
parser.add_argument('--m_real', action='store_true',default=False, help='')

# quantization 
parser.add_argument('--q_wbit', type=int, default=8)
parser.add_argument('--q_abit', type=int, default=8)
parser.add_argument('--q_clip',type=str, default='ft', choices=['max','mse','aciq','kl','ft'])
parser.add_argument('--q_lb', type=float, default=0.5, help='')
args = parser.parse_args()
misc.set_seed(args.seed)

if args.logger:
    args.timestr = time.strftime("%Y.%m.%d-%H%M%S")
    os.mkdir("./logs/"+args.timestr)
    logging.basicConfig(filename="./logs/"+args.timestr+'/log', filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
    logging.info(args)

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

# dataset
trainloader, testloader = dataset.get_dataset(args)

# generator
generator = models.generator.Generator(args).cuda()
if not args.m_real:
    generator.load_state_dict(misc.load_remove_parallel('./checkpoint/generator_'+args.t_model+'_'+args.dataset+'.pth'))

# teacher
teacher = models.get_model(args.t_model,args).cuda()
teacher.load_state_dict(torch.load('./checkpoint/' + args.t_model+'_'+args.dataset+'.pth'))

teacher.eval()
if not args.silent:
    acc, loss = test(teacher,testloader)
    print('==> Teacher Test, Loss: %f, Accuracy: %f' % (loss, acc))                        

if args.m_real:
    student = collect.collect_stats(teacher,None,trainloader,args).cuda()
else:
    student = collect.collect_stats(teacher,generator,None,args).cuda()

acc, loss = test(student,testloader)
print('==> %s, %s, W %d, A %d, clip: %s, lb: %.1f, Accuracy: %f' % (args.t_model, args.dataset,args.q_wbit, args.q_abit, args.q_clip,args.q_lb, acc))                        

def kdloss(output_s, output_t):
    p = F.log_softmax(output_s, dim=1)
    q = F.softmax(output_t, dim=1)
    l_kl = F.kl_div(p, q, reduction='sum') / output_s.shape[0]
    return l_kl

def FT(model_s,model_t,model_g,trainloader,testloader,args):
    if args.logger:
        tb_writer = SummaryWriter(log_dir='./logs/'+args.timestr)
        print('Logs in: ./logs/'+args.timestr)

    clip_param = filter(lambda p: p.shape[0]==1, model_s.parameters())
    base_param = filter(lambda p: p.shape[0]!=1, model_s.parameters())

    optimizer_S = torch.optim.SGD([{'params': base_param, 'lr':args.s_lr},
                                    {'params':clip_param, 'lr':args.s_lr/10,'momentum':0.,'weight_decay':0}],
                            lr=args.s_lr, momentum=0.9, weight_decay=5e-4)

    best_acc=test(model_s,testloader)[0]

    for epoch in range(args.s_epochs):
        if not args.silent:
            print('\nEpoch: %d' % epoch)
        model_s.train()

        train_loss_s = 0
        for i, (imgs, _) in enumerate(trainloader):
            if args.m_real:
                gen_imgs = imgs.cuda()
                outputs_T = model_t(gen_imgs)   
            else:
                with torch.no_grad():
                    z = Variable(torch.randn(args.batch_size, args.latent_dim)).cuda()
                    gen_imgs = model_g(z) 
                    outputs_T = model_t(gen_imgs)   


            optimizer_S.zero_grad()
            outputs_S = model_s(gen_imgs.detach())   
            loss_s = kdloss(outputs_S, outputs_T.detach()) 
            loss_s.backward()
            optimizer_S.step()

            train_loss_s += loss_s.item()

        test_result=test(model_s,testloader)
        if not args.silent:
            print('Test Loss: %.4f | Acc: %.4f | '% (test_result[1],test_result[0]))
        if args.logger:
            logging.info('[Epoch %d / %d] Train: Loss_S: %.4f | '% ( epoch, args.s_epochs, train_loss_s/(i+1)))
            logging.info('Test Loss: %.4f | Acc: %.4f | '% (test_result[0],test_result[1]))
            tb_writer.add_scalar('scalar/test_accuracy',test_result[0],epoch)

            if test_result[0] > best_acc:
                torch.save(model_s.state_dict(),'./logs/'+args.timestr+'/student.pth')
        best_acc = max(test_result[0], best_acc)

    if args.logger:
        tb_writer.close()
    print('DFQF best=',best_acc)

FT(student,teacher,generator,trainloader,testloader,args)

