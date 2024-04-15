# 10 crop for data enhancement
import torch.optim as optim
import numpy as np
import os
import argparse
import utils
from bfw_dataset import BFW
from models import *
import torchvision.transforms as transforms
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytz
from datetime import datetime

local_path = "/Users/h0z058l/Downloads/FER/codes/Face-Recognition/BFW"

parser = argparse.ArgumentParser(description='PyTorch BFW CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='BFW', help='CNN architecture')
parser.add_argument('--bs', default=64, type=int, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_test_acc = 0  # best Test accuracy
best_test_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 20  # 50
learning_rate_decay_every = 5  # 5
learning_rate_decay_rate = 0.9  # 0.9

total_epoch = 50

path = os.path.join(opt.dataset + '_' + opt.model)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((108, 124)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
# transform_test = transforms.Compose([
#     transforms.Lambda(lambda img: torch.stack([transforms.ToTensor()(img)])),
# ])
transform_test = transforms.Compose(
    [
        transforms.TenCrop(108), # 从108*124裁剪为108*108
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    ]
)

trainset = BFW(split='Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=0)
Testset = BFW(split='Test', transform=transform_test)
Testloader = torch.utils.data.DataLoader(Testset, batch_size=opt.bs, shuffle=False, num_workers=0)

# Model
if opt.model == 'VGG19':
    net = VGG('VGG19')
    print('==> Using VGG19 model..')
elif opt.model == 'Resnet18':
    net = ResNet18()
    print('==> Using ResNet18 model..')
elif opt.model == 'Resnet34':
    net = ResNet18()
    print('==> Using ResNet34 model..')

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path, 'Test_model.t7'))

    net.load_state_dict(checkpoint['net'])
    best_test_acc = checkpoint['Best_test_acc']
    best_test_acc_epoch = checkpoint['Best_test_acc_epoch']
    start_epoch = checkpoint['Best_test_acc_epoch'] + 1
else:
    print('==> Building model..')

if use_cuda:
    print('==> Using CUDA...')
    net.cuda()
else:
    print('==> No CUDA is available!')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

dallas_tz = pytz.timezone('America/Chicago')
local_time = time.localtime()
dallas_time = datetime.fromtimestamp(time.mktime(local_time), tz=pytz.utc).astimezone(dallas_tz)
filename = dallas_time.strftime("%m-%d-%H-%M-%S")
log_path = f"{local_path}/3/output/{filename}.txt"

outputfile = open(log_path, 'w')

loss_list = []
train_acc_list = []


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

    total_batches = len(trainloader)

    progress_bar = tqdm(range(total_batches // 100))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx % 100 == 0:
            progress_bar.update()
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

        if batch_idx % 100 == 0:
            print("Loss:", "{:.4e}".format(loss))
            print("Loss:", "{:.4e}".format(loss), file=outputfile)
        outputfile.flush()
        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    train_acc = 100. * correct / total
    print("Train_acc: %0.3f" % train_acc, file=outputfile)
    loss_list.append(train_loss)
    train_acc_list.append(train_acc)


test_acc_list = []


def test(epoch):
    global test_acc
    global best_test_acc
    global best_test_acc_epoch
    net.eval()
    test_loss = 0
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
        test_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(Testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    test_acc = 100. * correct / total
    test_acc_list.append(test_acc)

    if test_acc > best_test_acc:
        print("Best_test_acc: %0.3f" % test_acc, file=outputfile)

        state = {
            'net': net.state_dict() if use_cuda else net,
            'Best_test_acc': test_acc,
            'Best_test_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'Test_model.t7'))
        best_test_acc = test_acc
        best_test_acc_epoch = epoch


for epoch in range(start_epoch, total_epoch):
    train(epoch)
    test(epoch)

"""
------------------ plot figures --------------------
"""
x = range(0, 50)
print(train_acc_list, file=outputfile)
print(test_acc_list, file=outputfile)

plt.plot(x, train_acc_list, color='blue', label='Train_acc')
plt.plot(x, test_acc_list, color='orange', label='Test_acc')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.savefig(f"{local_path}/3/output/Accuracy-{filename}.png")
plt.show()

y3 = loss_list
plt.plot(x, y3, '-')
plt.title('Train_loss vs. epoches')
plt.xlabel('Epoch')
plt.ylabel('Train_loss')
plt.plot(x, y3, color='blue')
plt.savefig(f"{local_path}/3/output/Train-loss-{filename}.png")
plt.close()

outputfile.close()

print("Best_test_acc: %0.3f" % best_test_acc)
print("Best_test_acc_epoch: %d" % best_test_acc_epoch)
