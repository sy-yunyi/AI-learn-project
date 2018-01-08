#导入相关工具包

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator

def demo():
    dict = {
        'a': {'a': 1, 'b': 2},
        'd': {'f': 5, 'g': 8},
        'b': {'c': 3, 'd': 4}
    }
    d = {
        'a': 2,
        'c': 4,
        'b': 1
    }
    s = sorted(dict.items(), key=operator.itemgetter(1))
    print(s)

def demo_test():
    y=[12,13,34,23,45]
    people=['tom','Bob','Jack','Green','Mary']
    # plt.figure()
    # explode=[0.1,0.1,0.1,0.1,0.1]
    # plt.pie(y,labels=people,explode=explode,autopct='%1.1f%%')
    # plt.show()
    x=np.arange(len(people))
    ax = plt.gca()
    # ax.set_xticks(x)
    ax.set_xticklabels(people)
    plt.bar(x,y,facecolor='#9999ff',edgecolor='red')
    plt.show()



if __name__ == '__main__':
    demo_test()