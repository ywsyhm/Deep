#!/usr/bin/python
# -*- coding: UTF-8 -*-
__Author__ = "杨文升"
__version__ = '?'
"""This is the example module.

This module does stuff.
"""
import copy
import random
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from tqdm import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = r"\\models"


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))  # output(5, 30, 64)
        pred = self.linear(output)  # (5, 30, 1)
        pred = pred[:, -1, :]  # (5, 1)
        return pred


def set_seed(seed):
    torch.manual_seed(seed)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # gpu 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.num_directions * self.hidden_size, self.output_size)
    
    def forward(self, input_seq):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # print(input_seq.size())
        seq_len = input_seq.shape[1]
        # input(batch_size, seq_len, input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output)  # pred()
        pred = pred[:, -1, :]
        
        return pred


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, item):
        return self.data[item]
    
    def __len__(self):
        return len(self.data)


def load_data(file_name):
    df = pd.read_csv('D:\规上工业\LSTM预测\\' + file_name, encoding='gbk', index_col=0)
    columns = df.columns
    df.fillna(df.mean(), inplace=True)
    
    return df


def my_load_data(file_name, _input_size):
    df = pd.read_excel('D:\规上工业\LSTM预测\\' + file_name, index_col=[0, 1])
    columns = df.columns
    df.fillna(df.mean(), inplace=True)
    df = df.iloc[:, :_input_size]
    return df


def nn_seq_us(args):
    B = args.batch_size
    am10 = 0
    print('data processing...')
    dataset = load_data("data.csv")
    # split
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    m, n = np.max(train[train.columns[am10]]), np.min(train[train.columns[am10]])
    
    def process(data, batch_size, shuffle):
        load = data[data.columns[am10]]  # 选择使用数据集的第几列，在负荷预测中则是选择一天96个测定时间点中的一个
        load = load.tolist()
        data = data.values.tolist()
        load = [(x - n) / (m - n) for x in load]
        seq = []
        for i in range(len(data) - 24):
            train_seq = []
            train_label = []
            for j in range(i, i + 24):
                x = [load[j]]
                train_seq.append(x)
            # for c in range(2, 8):
            #     train_seq.append(data[i + 24][c])
            train_label.append(load[i + 24])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))
        
        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
        
        return seq
    
    Dtr = process(train, B, True)
    Val = process(val, B, True)
    Dte = process(test, B, False)
    
    return Dtr, Val, Dte, m, n


def nn_seq_ms(args):
    B = args.batch_size
    _input_size = args.input_size
    print('data processing...')
    dataset = load_data("mtl_data_1.csv")
    m, n = np.max(dataset[dataset.columns[0]]), np.min(dataset[dataset.columns[0]])
    scaler = MinMaxScaler()
    dataset = pd.DataFrame(scaler.fit_transform(dataset))
    # split
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    
    def process(data, batch_size, shuffle, _input_size):
        load = data[data.columns[0:3]]  # 选择使用数据集的第几列，在负荷预测中则是选择一天96个测定时间点中的一个
        load = load.values.tolist()
        # load = [(x - n) / (m - n) for x in load]
        data = data.values.tolist()
        # load = [(x - n) / (m - n) for x in load]
        seq = []
        for i in range(len(data) - 24):
            train_seq = []
            train_label = []
            for j in range(i, i + 24):
                x = load[j]
                train_seq.append(x)
            # for c in range(2, 8):
            #     train_seq.append(data[i + 24][c])
            train_label.append(load[i + 24][0])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))
        
        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
        
        return seq
    
    Dtr = process(train, B, True, _input_size)
    Val = process(val, B, True, _input_size)
    Dte = process(test, B, False, _input_size)
    
    return Dtr, Val, Dte, m, n


def my_nn_seq_ms(args, file_name):
    input_seq = args.input_seq
    B = args.batch_size
    _input_size = args.input_size
    print('data processing...')
    dataset = my_load_data(file_name, _input_size)
    m, n = np.max(dataset[dataset.columns[0]]), np.min(dataset[dataset.columns[0]])
    scaler = MinMaxScaler()
    dataset = pd.DataFrame(scaler.fit_transform(dataset))
    # split
    train = dataset[:int(len(dataset) * 0.7)]
    val = dataset[int(len(dataset) * 0.5):int(len(dataset) * 0.7)]
    test = dataset[int(len(dataset) * 0.7):len(dataset)]
    
    def process(data, batch_size, shuffle, _input_size):
        load = data[data.columns[0:_input_size]]  # 选择使用数据集的第几列，在负荷预测中则是选择一天96个测定时间点中的一个
        load = load.values.tolist()
        # load = [(x - n) / (m - n) for x in load]
        data = data.values.tolist()
        # load = [(x - n) / (m - n) for x in load]
        seq = []
        for i in range(len(data) - input_seq):
            train_seq = []
            train_label = []
            for j in range(i, i + input_seq):
                x = load[j]
                train_seq.append(x)
            # for c in range(2, 8):
            #     train_seq.append(data[i + 24][c])
            train_label.append(load[i + input_seq][0])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))
        
        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
        
        return seq
    
    Dtr = process(train, B, True, _input_size)
    Val = process(val, B, True, _input_size)
    Dte = process(test, B, False, _input_size)
    
    return Dtr, Val, Dte, m, n


def train(args, Dtr, Val, path):
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    
    loss_function = nn.MSELoss().to(device)
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
    min_val_loss = 5
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        # validation
        val_loss = loss  # get_val_loss(args, Model, Val)
        if epoch > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
        
        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()
    
    state = {'models': best_model.state_dict()}
    torch.save(state, path)
    return


def get_val_loss(args, model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(args.device)
    val_loss = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())
    
    return np.mean(val_loss)


def get_mape(y, pred):
    n = len(y)
    mape = sum([abs((y_temp - pred_temp) / y_temp) for y_temp, pred_temp in zip(y, pred)]) / n
    pass
    return mape


def test(args, Dte, path, m, n):
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    # models = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'], )
    model.eval()
    print('predicting...')
    for (seq, target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)
    
    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    # plot
    x = [i for i in range(1, len(y) + 1)]
    
    fig = plt.figure(dpi=900)
    plt.plot(x, y, c='green', marker='*', ms=1, alpha=0.75, label='true')
    plt.plot(x, pred, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    # plt.xticks(x, [5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6])
    plt.grid(axis='y')
    plt.legend()
    plt.show()
    return


def test_example(args, Dte, path, m, n):
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    # models = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)["models"])
    model.eval()
    print('predicting...')
    for (seq, target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)
    
    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    # plot
    x = [i for i in range(1, 151)]
    x_smooth = np.linspace(np.min(x), np.max(x), 900)
    y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
    fig = plt.figure(dpi=900)
    plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')
    y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.show()
    return
