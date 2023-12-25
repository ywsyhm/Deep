#!/usr/bin/python
# -*- coding: UTF-8 -*-
__Author__ = "杨文升"
__version__ = '?'
"""This is the example module.

This module does stuff.
"""

import argparse

import torch


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=3, help='number of batch of once input')
    parser.add_argument('--in_channels', type=int, default=[4, 128], help='input channels must be 4')
    parser.add_argument('--out_channels', type=int, default=[128, 256], help='the number of kernels')
    parser.add_argument('--kernel_size', type=int, default=[8, 6], )
    parser.add_argument('--output_size', type=int, default=1, help='output size')
    # parser.add_argument('--loss_function', type=int, default=1, help='output size')
    
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    
    parser.add_argument('--step_size', type=int, default=1, help='step size')
    parser.add_argument('--epochs', type=int, default=10, help='number of rounds of training')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay per global round')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    args = parser.parse_args()
    
    return args
