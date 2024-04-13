# 10 crop for data enhancement
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import numpy as np
import os
import argparse
import utils
from bfw_dataset import BFW
from torch.autograd import Variable
from models import *
import torchvision.transforms as transforms
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch BFW CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='BFW', help='CNN architecture')
parser.add_argument('--bs', default=128, type=int, help='learning rate')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_Test_acc = 0  # best Test accuracy
best_Test_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 50  # 50
learning_rate_decay_every = 5  # 5
learning_rate_decay_rate = 0.9  # 0.9

total_epoch = 10

path = os.path.join(opt.dataset + '_' + opt.model)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((108, 124)),  # 调整图片大小为 108x124
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.Lambda(lambda img: torch.stack([transforms.ToTensor()(img)])),
])

trainset = BFW(split='Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=0)
Testset = BFW(split='Test', transform=transform_test)
Testloader = torch.utils.data.DataLoader(Testset, batch_size=opt.bs, shuffle=False, num_workers=0)

# Model
if opt.model == 'VGG19':
    net = VGG('VGG19')
elif opt.model == 'Resnet18':
    net = ResNet18()

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path, 'Test_model.t7'))

    net.load_state_dict(checkpoint['net'])
    best_Test_acc = checkpoint['best_Test_acc']
    best_Test_acc_epoch = checkpoint['best_Test_acc_epoch']
    start_epoch = checkpoint['best_Test_acc_epoch'] + 1
else:
    print('==> Building model..')

if use_cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

filename = time.strftime("%m-%d-%H-%M-%S", time.localtime())
log_path = "/Users/h0z058l/Downloads/FER/codes/bfw/output/" + f'{filename}.txt'  # 也可以创建一个.doc的word文档
outputfile = open(log_path, 'w')

Loss_list = []
Train_acc_list = []


# Training
def train(epoch):
    print(f"Epoch: {epoch}", )
    print(f"Epoch: {epoch}", file=outputfile)

    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx % 25 == 0:
            print("Loss:", "{:.4e}".format(loss))
            print(f"Correct_cnt: {correct}")
            print("Loss:", "{:.4e}".format(loss), file=outputfile)
            print(f"Correct_cnt: {correct}", file=outputfile)
        outputfile.flush()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    Train_acc = 100. * correct / total


Test_acc_list = []


def Test(epoch):
    global Test_acc
    global best_Test_acc
    global best_Test_acc_epoch
    net.eval()
    Test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(Testloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        Test_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(Testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (Test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    Test_acc = 100. * correct / total

    if Test_acc > best_Test_acc:
        print('Saving..')
        print("best_Test_acc: %0.3f" % Test_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'best_Test_acc': Test_acc,
            'best_Test_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'Test_model.t7'))
        best_Test_acc = Test_acc
        best_Test_acc_epoch = epoch


for epoch in range(start_epoch, total_epoch):
    train(epoch)
    Test(epoch)

"""
------------------ plot figures --------------------
"""
x1 = range(0, 10)
y1 = Train_acc_list
print(y1)
print(y1, file=outputfile)
plt.plot(x1, y1, '-')
plt.title('Train_acc vs. epoches')
plt.ylabel('Train_acc')
plt.plot(x1, y1, color='blue', label='Train_acc')
plt.savefig(f"/Users/h0z058l/Downloads/FER/codes/bfw/output/Train-acc-{filename}.png")

x2 = range(0, 10)
y2 = Loss_list
print(y2)
plt.plot(x2, y2, '-')
plt.title('Train_loss vs. epoches')
plt.xlabel('Epoch')
plt.ylabel('Train_loss')
plt.plot(x2, y2, color='blue')
plt.savefig(f"/Users/h0z058l/Downloads/FER/codes/bfw/output/Train-loss-{filename}s.png")

x3 = range(0, 10)
y3 = Test_acc_list
print(y3)
print(y3, file=outputfile)
plt.plot(x3, y3, '-')
plt.title('Test_acc vs. epoches')
plt.xlabel('Epoch')
plt.ylabel('Test_acc')
plt.plot(x3, y3, color='orange', label='Private_acc')
plt.legend(loc='lower right')
plt.savefig(f"/Users/h0z058l/Downloads/FER/codes/bfw/output/Test-acc-{filename}.png")
plt.close()

outputfile.close()
print("best_Test_acc: %0.3f" % best_Test_acc)
print("best_Test_acc_epoch: %d" % best_Test_acc_epoch)
