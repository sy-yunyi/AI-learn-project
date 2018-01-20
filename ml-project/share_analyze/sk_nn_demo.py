# 量化交易，Kmeans


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tushare as ts
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pdb
from sklearn.neural_network import MLPClassifier


object_trade = '000001'

def get_xy(start_date,end_date):
    """
    获得数据集和分类
    :param start_date: 开始时间
    :param end_date: 结束时间
    :return: 数据集和增益
    """
    data = ts.get_k_data(object_trade,index=True,start=start_date,end=end_date)

    closes = data['close'].values
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values

    re_x =[]
    re_y = []

    for i in range(2,len(opens)):
        tmp = closes[i-2]                   # 加入标线，
        re_x.append([opens[i-1]/tmp,highs[i-1]/tmp,lows[i-1]/tmp,closes[i-1]/tmp])    #加入今天的特征值
        re_y.append(closes[i]/closes[i-1]-1)     # 使用明天收盘价/今天的收盘价     ---- 计算增益，可能减少或者增加
    x= np.array(re_x)
    y = np.array(re_y)
    return x,y


def fig(labels,labels_u,re_y):
    """

    :param labels: 分类机预测值
    :param labels_u: 不重复的标签集
    :param re_y: 增益
    :return: None
    """
    r_list = []    #增益求和
    titles = []
    for i in range(len(labels_u)):
        tem = labels == labels_u[i]        #判断分类，
        r_list.append(re_y[tem])            #取出所有分类正确的
        titles.append('Labels:'+str(labels_u[i]))
    plt.figure()
    for i in range(len(titles)):
        plt.plot(r_list[i].cumsum(),label=titles[i])
    plt.legend()     #添加标注
    plt.grid()         # 画出表格
    plt.show()




x,y = get_xy('2000-01-01','2013-01-01')
tmp = np.ones(len(y))
tmp[y > 0.005] = 2
tmp[y < -0.005] = 0
scaler = preprocessing.StandardScaler()
x =scaler.fit_transform(x)

pca = PCA(3)
x = pca.fit_transform(x)

# pdb.set_trace()
# solver(修改误差方法) :sgd(一定程度上跳出局部最优),adma(数据大),lbfgs(数据少)
clf = MLPClassifier(solver='adam', max_iter=2000, hidden_layer_sizes=(10,5),activation='tanh')
kk = clf.fit(x,tmp)


labels = clf.predict(x)
labels_u = np.unique(labels)
# fig(labels,labels_u,y)

x,y = get_xy('2013-01-01','2018-01-01')

x = scaler.transform(x)
x = pca.transform(x)
labels = clf.predict(x)
fig(labels,labels_u,y)
