from sklearn.neighbors import KNeighborsClassifier
import tushare as ts
import numpy as np
import matplotlib.pyplot as plt


object_trade = '000001'

def get_xy(start_date,end_date):
    data = ts.get_k_data(object_trade,index=True,start=start_date,end=end_date)
    closes = data['close'].values
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values

    re_x =[]
    re_y = []

    for i in range(2,len(opens)):
        tmp = closes[i-2]                   # 加入标线，
        re_x.append([opens[i-1]/tmp,highs[i-1]/tmp,lows[i-1]/tmp,closes[i-1]/tmp])
        re_y.append(closes[i]/closes[i-1]-1)
    x= np.array(re_x)
    y = np.array(re_y)
    return x,y

x,y = get_xy('2009-01-01','2016-01-01')
# 调用sklearn的kNN，
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x,y>0)

x,y = get_xy('2016-01-01','2018-01-01')
labels = knn.predict(x)
tmp = y[labels>0]
plt.grid()
plt.plot(tmp.cumsum())
plt.show()



# def get_dis(inx,iny):
#     return (((inx-iny)**2).sum())**0.5

# x = np.array([3417,3442,3404,3430])
# y=np.array([3314,3349, 3314, 3348])
# z= np.array([3403,3437 ,3401 ,3436])
# dis1=get_dis(x,z)
# dis2=get_dis(y,z)

# print(dis1,dis2)