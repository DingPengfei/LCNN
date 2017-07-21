# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.io as sio
import numpy as np
import itertools

# Hyper Parameters
num_epochs = 5
batch_size = 512
learning_rate = 0.0001

# read data
# @Author: Lv Cheng
np.random.seed(10)
Dimension = 1900
single_sample = 560
Samples = 44 * 560 + 29  # 24669
totalIter = 10000+1
trainX = np.zeros([Samples - single_sample, 8, Dimension])
trainY = np.zeros([Samples - single_sample, 1])
testX = np.zeros([single_sample, 8, Dimension])
testY = np.zeros([single_sample, 1])


def read_data():
    global trainX, trainY, testX, testY

    X = np.zeros([Samples, 8, Dimension])
    Label = np.zeros([Samples, 1])

    path = './data/'
    for i in range(44):
        filename = 'jinshanwhole' + str(i).zfill(2) + '.mat'

        mat = sio.loadmat(path + filename)['ecgBeatsInfo']
        # block_size = len(mat)	# 0~43:560, 44:29

        st = i * 560
        en = st
        if (i < 44):
            en += 560
        else:
            en += 29

        X[st:en] = mat[:, 5:].reshape(-1, 8, Dimension)  # block_size * 8 * 1900
        Label[st:en] = mat[:, 4].reshape(-1, 1)  # block_size * 1

    print ("shape of X:", X.shape)

    # split the dataset
    trainX = X[single_sample:, :, :]
    trainY = Label[single_sample:, :]

    testX = X[:single_sample, :, :Dimension - 200]
    testY = Label[:single_sample, :]

    print ("Finish reading.")


def next_batch(batch_size=128):
    # should be integer of 0~42
    index = np.random.randint(43)
    # print ("train set is ", i)
    batch_xs = np.zeros([batch_size, 8, Dimension-200, 1])
    batch_ys = np.zeros([batch_size, 1])

    for i in range(batch_size):
        type = np.random.randint(2)
        if type == 1:
            # 1 ~ 160
            x = np.random.randint(160)+1
            start = np.random.randint(200)

            batch_xs[i] = trainX[index*560+x, :, start:start+Dimension-200].reshape(-1, 8, Dimension-200, 1)
            batch_ys[i] = trainY[index*560+x]
        else:
            # 161 ~ 560
            x = np.random.randint(400)+161
            start = np.random.randint(200)

            batch_xs[i] = trainX[index*560+x, :, start:start+Dimension-200].reshape(-1, 8, Dimension-200, 1)
            batch_ys[i] = trainY[index*560+x]

    return batch_xs, batch_ys


# LCNN Model
class LCNN(nn.Module):
    # 定义Net的初始化函数，这个函数定义了该神经网络的基本结构
    def __init__(self):

        super(LCNN, self).__init__()

        self.conv1_1 = nn.Conv1d(1, 6, 21)
        self.conv1_2 = nn.Conv1d(1, 6, 21)
        self.conv1_3 = nn.Conv1d(1, 6, 21)
        self.conv1_4 = nn.Conv1d(1, 6, 21)
        self.conv1_5 = nn.Conv1d(1, 6, 21)
        self.conv1_6 = nn.Conv1d(1, 6, 21)
        self.conv1_7 = nn.Conv1d(1, 6, 21)
        self.conv1_8 = nn.Conv1d(1, 6, 21)

        self.conv2_1 = nn.Conv1d(6, 7, 13)
        self.conv2_2 = nn.Conv1d(6, 7, 13)
        self.conv2_3 = nn.Conv1d(6, 7, 13)
        self.conv2_4 = nn.Conv1d(6, 7, 13)
        self.conv2_5 = nn.Conv1d(6, 7, 13)
        self.conv2_6 = nn.Conv1d(6, 7, 13)
        self.conv2_7 = nn.Conv1d(6, 7, 13)
        self.conv2_8 = nn.Conv1d(6, 7, 13)

        self.conv3_1 = nn.Conv1d(7, 5, 9)
        self.conv3_2 = nn.Conv1d(7, 5, 9)
        self.conv3_3 = nn.Conv1d(7, 5, 9)
        self.conv3_4 = nn.Conv1d(7, 5, 9)
        self.conv3_5 = nn.Conv1d(7, 5, 9)
        self.conv3_6 = nn.Conv1d(7, 5, 9)
        self.conv3_7 = nn.Conv1d(7, 5, 9)
        self.conv3_8 = nn.Conv1d(7, 5, 9)

        self.fc1 = nn.Linear(8*5*5, 50)

        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):

        # xx1 = np.transpose(x[0])
        # xx2 = np.transpose(x[0][0])
        # xx3 = np.transpose(x[0][0][0])
        # xx4 = torch.from_numpy(np.transpose(x[0]))
        # xx4 = torch.from_numpy(np.transpose(x[0])).float()
        x1 = Variable((torch.from_numpy(np.transpose(x[0]))).unsqueeze(0).float())
        x2 = Variable((torch.from_numpy(np.transpose(x[1]))).unsqueeze(0).float())
        x3 = Variable((torch.from_numpy(np.transpose(x[2]))).unsqueeze(0).float())
        x4 = Variable((torch.from_numpy(np.transpose(x[3]))).unsqueeze(0).float())
        x5 = Variable((torch.from_numpy(np.transpose(x[4]))).unsqueeze(0).float())
        x6 = Variable((torch.from_numpy(np.transpose(x[5]))).unsqueeze(0).float())
        x7 = Variable((torch.from_numpy(np.transpose(x[6]))).unsqueeze(0).float())
        x8 = Variable((torch.from_numpy(np.transpose(x[7]))).unsqueeze(0).float())

        x1 = F.max_pool1d(F.relu(self.conv1_1(x1)), 7)
        x2 = F.max_pool1d(F.relu(self.conv1_2(x2)), 7)
        x3 = F.max_pool1d(F.relu(self.conv1_3(x3)), 7)
        x4 = F.max_pool1d(F.relu(self.conv1_4(x4)), 7)
        x5 = F.max_pool1d(F.relu(self.conv1_5(x5)), 7)
        x6 = F.max_pool1d(F.relu(self.conv1_6(x6)), 7)
        x7 = F.max_pool1d(F.relu(self.conv1_7(x7)), 7)
        x8 = F.max_pool1d(F.relu(self.conv1_8(x8)), 7)

        x1 = F.max_pool1d(F.relu(self.conv2_1(x1)), 6)
        x2 = F.max_pool1d(F.relu(self.conv2_2(x2)), 6)
        x3 = F.max_pool1d(F.relu(self.conv2_3(x3)), 6)
        x4 = F.max_pool1d(F.relu(self.conv2_4(x4)), 6)
        x5 = F.max_pool1d(F.relu(self.conv2_5(x5)), 6)
        x6 = F.max_pool1d(F.relu(self.conv2_6(x6)), 6)
        x7 = F.max_pool1d(F.relu(self.conv2_7(x7)), 6)
        x8 = F.max_pool1d(F.relu(self.conv2_8(x8)), 6)

        x1 = F.max_pool1d(F.relu(self.conv3_1(x1)), 6)
        x2 = F.max_pool1d(F.relu(self.conv3_2(x2)), 6)
        x3 = F.max_pool1d(F.relu(self.conv3_3(x3)), 6)
        x4 = F.max_pool1d(F.relu(self.conv3_4(x4)), 6)
        x5 = F.max_pool1d(F.relu(self.conv3_5(x5)), 6)
        x6 = F.max_pool1d(F.relu(self.conv3_6(x6)), 6)
        x7 = F.max_pool1d(F.relu(self.conv3_7(x7)), 6)
        x8 = F.max_pool1d(F.relu(self.conv3_8(x8)), 6)

        x1 = x1.view(-1, self.num_flat_features(x1))
        x2 = x2.view(-1, self.num_flat_features(x2))
        x3 = x3.view(-1, self.num_flat_features(x3))
        x4 = x4.view(-1, self.num_flat_features(x4))
        x5 = x5.view(-1, self.num_flat_features(x5))
        x6 = x6.view(-1, self.num_flat_features(x6))
        x7 = x7.view(-1, self.num_flat_features(x7))
        x8 = x8.view(-1, self.num_flat_features(x8))

        # combine 8 leads
        x = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8], dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

lcnn = LCNN()
lcnn.float()
read_data()

# Loss and Optimizer
losses = []
criterion = nn.BCELoss()
optimizer = optim.Adam(lcnn.parameters(), lr=learning_rate)
scheduler = .ExponentialLR(optimizer, gamma=0.1)

# Train the Model
for epoch in range(num_epochs):
    batch_xs, batch_ys = next_batch(batch_size)
    total_loss = torch.Tensor([0])
    i = 1
    for data, target in itertools.izip(batch_xs, batch_ys):
        # Forward + Backward + Optimize
        lcnn.zero_grad()
        # optimizer.zero_grad() ???
        outputs = lcnn(data)
        target = Variable(torch.from_numpy(target).float())
        loss = criterion(outputs, target)
        loss.backward()
        scheduler.step()
        total_loss += loss.data

        print ('Epoch [%d/%d] Iter [%d/%d] Loss: %.4f'
               % (epoch + 1, num_epochs, i, len(batch_xs), loss.data[0]))
        i = i+1
    losses.append(total_loss)

# Test the Model
