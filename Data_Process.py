#!/usr/bin/python
# -*- coding: UTF-8 -*-
__Author__ = "杨文升"
__version__ = '?'

"""This is the example module.

This module does stuff.
"""
import os
import sys

import numpy as np

factor = 1


def seq_matrix(seq_list, dim):  # One Hot Encoding
    # 转换全部字母序列变为，三维向量
    
    tensor = np.zeros((len(seq_list), dim, 4))
    
    for i in range(len(seq_list)):
        seq = seq_list[i]
        j = 0
        for s in seq:
            if s == 'A' or s == 'a':
                tensor[i][j] = [1, 0, 0, 0]
            if s == 'T' or s == 't':
                tensor[i][j] = [0, 1, 0, 0]
            if s == 'C' or s == 'c':
                tensor[i][j] = [0, 0, 1, 0]
            if s == 'G' or s == 'g':
                tensor[i][j] = [0, 0, 0, 1]
            if s == 'N':
                tensor[i][j] = [0, 0, 0, 0]
            j += 1
    return tensor


def fasta_to_matrix(dirPath):
    global Data
    Data = {}
    dim = 2000
    len_pos = 0
    len_neg = 0
    negName = "neg" + str(factor) + "X"
    for root, dirs, fileNames in os.walk(dirPath):
        for fileName in fileNames:
            
            if "pos" not in fileName and negName not in fileName:
                continue
            print(fileName + r' starting!')
            
            seq = []
            key = "NA"
            lines = loadfile(os.path.join(root, fileName))
            if "pos" in fileName:
                key = "pos"
                len_pos = len(lines)
            elif negName in fileName:
                key = "neg"
                len_neg = len(lines)
            else:
                continue
            
            Data[key] = seq_matrix(lines, dim)
            print(fileName + r' end!')
        
        if len_pos * len_neg == 0:
            raise IOError("缺少输入文件！")
        Data_Test = np.concatenate([Data["pos"], Data["neg"]])  # Test data
        Label_Test = np.concatenate([np.ones(len_pos), np.zeros(len_neg)])  # Test label
        
        np.save('Data_Processed/Data_processed' + str(factor) + 'X', Data_Test)
        np.save('Data_Processed/Label_processed' + str(factor) + 'X', Label_Test)
        break


def loadfile(filePath):
    with open(filePath) as f:
        ls = []
        for line in f:
            if not line.startswith('>'):
                ls.append(line.replace('\n', ''))  # 去掉行尾的换行符真的很重要！
    
    return ls


if __name__ == '__main__':
    fasta_to_matrix("./Data")
    print('Done')
sys.exit(0)
