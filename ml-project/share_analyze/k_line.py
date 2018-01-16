import tushare as ts
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
from matplotlib.pylab import date2num
import numpy as np
import datetime


data = ts.get_k_data('600782',start='2017-09-01',end='2018-01-01')
dates = data['date'].values

price = data[['open','high','low','close']]


# dates = [date2num(datetime.datetime.strptime(x,'%Y-%m-%d')) for x in dates]
candle_data = np.column_stack([list(range(len(dates))),price])



l = len(dates)
xindex = list(range(0,l,l//12))

# candle_data = np.column_stack([dates,price])
fig = plt.figure()

ax = fig.add_axes([0.1,0.3,0.8,0.6])
mpf.candlestick_ohlc(ax,candle_data,width=0.5,colorup='r',colordown='g')   # 画出k线图

closes = data['close']
ma5 = []
ma10 = []
for i in range(4,l):
    ma5.append(np.mean(closes[i-4:i+1]))
    if i >= 9:
        ma10.append(np.mean(closes[i-9:i+1]))
ax.plot(range(4,l),ma5)
ax.plot(range(9,l),ma10)
ax.grid()
ax.set_xticks(xindex)
ax1 = fig.add_axes([0.1,0.1,0.8,0.2])

ax1.bar(range(l),data['volume'])

# ax.xaxis_date()  #设置X轴为日期


ax1.set_xticks(xindex)  # 对选中的点定位
ax1.set_xticklabels(dates[xindex],rotation = 30)


ax1.grid()  #加方格线
# plt.xticks(rotation = 30)
plt.show()