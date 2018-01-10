import numpy as np
import pandas as pd
from math import log

def create_data_set():
    data_set=[[1, 1, 'yes'],[1,1,'yesy'],[1, 0,'no' ],[0, 1, 'no'],[0, 1, 'no']]
    lables=['no surfacing','surfacing']
    return data_set,lables

def calcShannonEnt(dataSet):

    numEntries=len(dataSet)
    lablesCounts={}
    for i_vector in dataSet:
        currentLable=i_vector[-1]
        lablesCounts[i_vector]=lablesCounts.get(i_vector,0)+1
    shannonEnt=0.0
    for key in lablesCounts:
        prob= float(lablesCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt


if __name__ == '__main__':
    data_set,lables=create_data_set()
    print(calcShannonEnt(data_set))