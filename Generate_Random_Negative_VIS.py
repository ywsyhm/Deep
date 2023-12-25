#!/usr/bin/python
# -*- coding: UTF-8 -*-
__Author__ = "杨文升"
__version__ = '?'
"""
生成不同规模的阴性数据位点
"""
import sys
from itertools import accumulate

import numpy as np
import pandas as pd

np.random.seed(7)
factor = 1
chr_serial = ["NC_060925.1", "NC_060926.1", "NC_060927.1", "NC_060928.1", "NC_060929.1", "NC_060930.1", "NC_060931.1",
              "NC_060932.1", "NC_060933.1", "NC_060934.1", "NC_060935.1", "NC_060936.1", "NC_060937.1", "NC_060938.1",
              "NC_060939.1", "NC_060940.1", "NC_060941.1", "NC_060942.1", "NC_060943.1", "NC_060944.1", "NC_060945.1",
              "NC_060946.1", "NC_060947.1", "NC_060948.1"]
if __name__ == '__main__':
    ChrStat = pd.read_csv('Data/ChrStat.txt', sep="\t")
    HBVIS = pd.read_csv('Data/HBVIS_location.txt', sep="\t")
    ChrLengths = ChrStat['Length'].to_list()
    ChrLengthsSum = sum(ChrLengths)
    ChrAccLength = list(accumulate(ChrLengths))
    ChrAccLength.insert(0, 0)
    HBVIS_Num = len(HBVIS)
    for line in range(HBVIS_Num):
        HBVIS.loc[line, 'AccLocation'] = int(ChrAccLength[HBVIS.loc[line, "Chr"] - 1] + HBVIS.loc[line, 'Location'])
    HBVIS.loc[HBVIS_Num] = [1, 0, 0]
    HBVIS.loc[HBVIS_Num + 1] = [24, 3117275502, 3117275502]
    HBVIS = HBVIS.astype('int64')
    HBVIS = HBVIS.sort_values(by="AccLocation", ascending=True)
    
    # 生成随机数
    size = factor * HBVIS_Num
    RawRandom = np.random.randint(1, ChrAccLength[-1], size=int(10 * factor * HBVIS_Num), dtype=np.int64)
    
    randomNum = 0
    FinalRandom = []
    try:
        for ran in RawRandom:
            i = np.searchsorted(HBVIS['AccLocation'], ran)
            distance1 = ran - HBVIS.loc[i - 1, 'AccLocation']
            distance2 = HBVIS.loc[i, 'AccLocation'] - ran
            if distance1 > 50000 and distance2 > 50000:
                FinalRandom.append(ran)
                randomNum += 1
            if randomNum == size:
                break
    except:
        print(ran, i)
    ChromosomeList = []
    LocationList = []
    for ran_final in FinalRandom:
        i_final = np.searchsorted(ChrAccLength, ran_final)
        chromosome_final = chr_serial[i_final - 1]
        location_final = ran_final - ChrAccLength[i_final - 1]
        ChromosomeList.append(chromosome_final)
        LocationList.append(location_final)
    Result = pd.DataFrame({'Chr': ChromosomeList, 'Location': LocationList})
    
    Result.to_csv(r'./Data/HBVIS_random.txt', header=False, sep='\t', index=False)
    print("Done")
sys.exit(0)
