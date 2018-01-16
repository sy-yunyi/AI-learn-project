# 在开盘是买入，收盘时卖出策略
# 还需要考虑手续费
# 一般为千分之一点三
import tushare as ts
import numpy as np
import matplotlib.pyplot as plt
import pdb
stock = '000001'
data = ts.get_k_data(stock,start = '2010-01-01',end = '2018-01-01')
opens = data['open'].values
closes = data['close'].values


diff = []
for i in range(2,len(closes)):
    if closes[i-1]/closes[i-2] <1.094 and 1-closes[i-1]/closes[i-2]<0.094 and closes[i-1]/closes[i-2]<=1:   #去除涨跌停
        diff.append(1-opens[i]/closes[i])
diff = np.array(diff)
# pdb.set_trace()
plt.plot(diff.cumsum())
plt.show()
