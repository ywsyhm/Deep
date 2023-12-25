#!/usr/bin/python
# -*- coding: UTF-8 -*-
__Author__ = "杨文升"
__version__ = '?'

"""This is the example module.

This module does stuff.
"""
import os
import sys

from CNN import *
from args import args_parser

# set_seed(200)
path = os.getcwd()
MODEL_PATH = path + r'\\Model\\CNN.pkl'

if __name__ == '__main__':
    args = args_parser()
    set_seed(1)
    # Dtr, Val, Dte, m, n = nn_seq_us(args)
    # train(args, Dtr, Val, path)
    Dtr, Val, Dte = my_nn_seq_ms(args, "Data_processed1X.npy", "Label_processed1X.npy")
    train(args, Dtr, Val, MODEL_PATH)
    #
    # # test(args, Dtr, LSTM_PATH, m, n)
    # test(args, Dtr, LSTM_PATH, m, n)
    # test(args, Dte, LSTM_PATH, m, n)
    
    sys.exit(0)
