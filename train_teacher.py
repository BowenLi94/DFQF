import os
import torch
from torch.autograd import Variable
import torch.optim as optim

import argparse

import models
from utils import dataset

parser = argparse.ArgumentParser(description='train-teacher-network')

# Basic model parameters.
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','cifar100'])
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--model', type=str, default='resnet20')
parser.add_argument('--output_dir', type=str, default='./checkpoint/')
parser.add_argument('--seed', type=int, default=1, help='random seed')

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

os.makedirs(args.output_dir, exist_ok=True)  

acc_best = 0
data_train_loader, data_test_loader = dataset.get_dataset(args)

net = models.get_model(args.model,args).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [80,160], gamma=0.1)


def train(epoch):
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
 
        optimizer.zero_grad()
 
        output = net(images)
 
        loss = criterion(output, labels)
 
        loss_list.append(loss.data.item())
        batch_list.append(i+1)
 
        if i == 1:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))
 
        loss.backward()
        optimizer.step()
 
def test():
    global  acc_best
    net.eval()
    correct = 0.0
    total = 0.0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
            total += labels.size(0)
 
    avg_loss /= i+1
    acc = float(correct)/total
    if acc_best < acc:
        acc_best = acc
        torch.save(net.state_dict(),args.output_dir + '/' +args.model+'_'+args.dataset+'.pth')
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
  
def train_and_test(epoch):
    train(epoch)
    test()
    lr_scheduler.step(epoch)

def main():
    for epoch in range(1, args.epochs):
        train_and_test(epoch)
 
 
if __name__ == '__main__':
    main()
