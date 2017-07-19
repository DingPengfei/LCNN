# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.io as sio
from os import listdir
from os.path import isfile, join
import numpy as np


# mypath = './data/'
# files = [f for f in listdir(mypath) if f != 'jinshanwhole00.mat']
# test = 'jinshanwhole00.mat'
# size = len(files)
#
# for i in range(size):
#     mat = sio.loadmat(mypath + files[i])['ecgBeatsInfo']
#     mat_size = mat.shape[0]
#     mats = [f.tolist() for f in mat[:, ]]
#     normal = [f]
#     # abnormal = [f for f in mat[:, ] if f[1, 5] == 1]
#     pass




# @Author: Lv Cheng
np.random.seed(10)
Dimension = 1900
single_sample = 560
Samples = 44 * 560 + 29 # 24669

totalIter = 10000+1

trainX = np.zeros([Samples - single_sample, 8, Dimension])
trainY = np.zeros([Samples - single_sample, 1])
testX = np.zeros([single_sample, 8, Dimension])
testY = np.zeros([single_sample, 1])

conv1_size = 21; conv1_feature = 6; max_pool1_size = 7
conv2_size = 13; conv2_feature = 7; max_pool2_size = 6
conv3_size =  9; conv3_feature = 5; max_pool3_size = 6

fc_connect_size = 50

def ReadDataSet():
    global trainX, trainY, testX, testY

    X = np.zeros([Samples, 8, Dimension])
    Label = np.zeros([Samples, 1])

    path = './data/'
    for i in range(44):
        filename = 'jinshanwhole' + str(i).zfill(2) + '.mat'

        # ecgBeatsInfo shape : 560 * 15205
        # 15205 = 0~0 (order) +
        # 1~1 (constant：1) +
        # 2~2 (constant: 8) +
        # 3~3 (constant: 1900) +
        # 4~4 (classification 0:Normal 1:Abnormal) +
        # 5~15204 = 8(II, III, V1, V2, V3, V4, V5, V6) * 1900

        mat = sio.loadmat(path + filename)['ecgBeatsInfo']
        # block_size = len(mat)	# 0~43:560, 44:29

        st = i * 560;
        en = st
        if (i < 44):
            en += 560
        else:
            en += 29

        X[st:en] = mat[:, 5:].reshape(-1, 8, Dimension)  # block_size * 8 * 1900
        Label[st:en] = mat[:, 4].reshape(-1, 1)  # block_size * 1

    '''
    # memory overflow
    mean = X.mean(axis=2); std = X.std(axis=2)
    X = (X - mean[:,:,np.newaxis])/(std[:,:,np.newaxis] + 1e-6)
    '''
    print ("shape of X:", X.shape)
    '''
    mean = X.mean(axis=2); std = X.std(axis=2)
    for i in range(np.shape(X)[0]):
        for j in range(np.shape(X)[1]):
            for k in range(np.shape(X)[2]):
                X[i][j][k] = (X[i][j][k] - mean[i][j]) / (std[i][j] + 1e-6)
    '''

    # split the dataset
    trainX = X[single_sample:, :, :]
    trainY = Label[single_sample:, :]

    testX = X[:single_sample, :, :Dimension - 200]
    testY = Label[:single_sample, :]

    print ("Finish reading.")




class Net(nn.Module):
    # 定义Net的初始化函数，这个函数定义了该神经网络的基本结构
    def __init__(self):

        super(Net, self).__init__()

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


        self.fc1 = nn.Linear(16*5*5, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):

        x = F.max_pool1d(F.relu(self.conv1(x)), 7)

        x = F.max_pool1d(F.relu(self.conv1(x)), 6)

        x = F.max_pool1d(F.relu(self.conv1(x)), 6)

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x

    # 使用num_flat_features函数计算张量x的总特征量（把每个数字都看出是一个特征，即特征总量），比如x是4*2*2的张量，那么它的特征总量就是16。
    def num_flat_features(self, x):
        # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片，那么输入数据张量的维度自然上升到了4维。【1:】让我们把注意力放在后3维上面
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

ReadDataSet()
net = Net()
net

