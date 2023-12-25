#!/usr/bin/python
# -*- coding: UTF-8 -*-
__Author__ = "杨文升"
__version__ = '?'
"""This is the example module.

This module does stuff.
"""
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from tqdm import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = r"./Models"


def set_seed(seed):
    torch.manual_seed(seed)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # gpu 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, batch_size):
        super(CNN, self).__init__()  # 继承__init__功能
        ## 第一层卷积
        self.batch_size = batch_size
        self.conv1 = nn.Sequential(
            # 输入[1,4,2000]
            nn.Conv1d(
                in_channels=in_channels[0],  # 输入图片的高度
                out_channels=out_channels[0],  # 输出图片的高度
                kernel_size=kernel_size[0],  # 5x5的卷积核，相当于过滤器
                stride=1,  # 卷积核在图上滑动，每隔一个扫一次
                padding=0,  # 给图外边补上0
            ),
            # 经过卷积层 输出[1,128,1993] 传入池化层
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)  # 经过池化 输出[1,128,664] 传入下一个卷积
        )
        ## 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels[1],  # 输入图片的高度
                out_channels=out_channels[1],  # 输出图片的高度
                kernel_size=kernel_size[1],
                stride=1,
                padding=0
            ),
            # 经过卷积 输出[1, 256, 659] 传入池化层
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)  # 经过池化 输出[1,256,219] 传入输出层
        )
        ## 输出层
        self.output = nn.Linear(in_features=1 * 256 * 219, out_features=1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)  #
        x = x.view(x.size(0), -1)  # 保留batch
        output = self.output(x)
        # output = torch.sigmoid(output)
        return output


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, item):
        return self.data[item]
    
    def __len__(self):
        return len(self.data)


def my_load_data(file_name, ):
    npy = np.load(r'Data_Processed\\' + file_name, )
    # columns = df.columns
    
    return npy


def get_val_loss(args, model, Val):
    model.eval()
    loss_function = nn.CrossEntropyLoss().to(args.device)
    val_loss = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred.squeeze(-1), label)
            val_loss.append(loss.item())
    
    return np.mean(val_loss)


def my_nn_seq_ms(args, seq_file_name, label_file_name):
    B = args.batch_size
    
    print('\ndata processing...')
    seq = my_load_data(seq_file_name, )
    label = my_load_data(label_file_name, )
    
    # Determine the number of positive
    pos_num = int(sum(label))
    
    # split
    seq_pos = seq[:pos_num]
    seq_neg = seq[pos_num:]
    
    def process(_seq_pos, _seq_neg, _type, batch_size, shuffle, ):
        type_split = {"train": [_seq_pos[:int(len(_seq_pos) * 0.02)],
                                _seq_neg[:int(len(_seq_neg) * 0.02)]],
                      "validate": [_seq_pos[int(len(_seq_pos) * 0.5):int(len(_seq_pos) * 0.52)],
                                   _seq_neg[int(len(_seq_neg) * 0.5):int(len(_seq_neg) * 0.52)]],
                      "test": [_seq_pos[int(len(_seq_pos) * 0.7):],
                               _seq_neg[int(len(_seq_neg) * 0.7):]]
                      }
        seq1 = type_split[_type][0]
        seq2 = type_split[_type][1]
        _seq = torch.FloatTensor(seq1 + seq2)
        
        _label = np.concatenate([np.ones(len(seq1)), np.zeros(len(seq2))])
        _label = torch.FloatTensor(_label)
        seq_label = []
        for i in range(len(_seq)):
            seq_label.append((_seq[i], _label[i]))
        seq_label = MyDataset(seq_label)
        _seq = DataLoader(dataset=seq_label, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
        return _seq
    
    Dtr = process(seq_pos, seq_neg, 'train', B, True)
    Val = process(seq_pos, seq_neg, "validate", B, True)
    Dte = process(seq_pos, seq_neg, "test", B, False)
    
    return Dtr, Val, Dte


def train(args, Dtr, Val, path):
    in_channels, out_channels, kernel_size = args.in_channels, args.out_channels, args.kernel_size
    
    model = CNN(in_channels, out_channels, kernel_size, batch_size=args.batch_size).to(device)
    
    loss_function = nn.CrossEntropyLoss().to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 5
    best_model = None
    min_val_loss = 10000000
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred.squeeze(-1), label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        # validation
        val_loss = get_val_loss(args, model, Val)
        if epoch > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
            # print(best_model)
        
        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()
    
    state = {'models': best_model.state_dict()}
    torch.save(state, path)
    return
