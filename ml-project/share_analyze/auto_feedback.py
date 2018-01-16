import numpy as np
import pdb
import matplotlib.pyplot as plt
import tushare  as ts
import win32api
import time


stock =['600000','601326','601949']
traget_price = [12.87,6.20,7,24]
while(1):
    time.sleep(1)
    price_current = [float(ts.get_realtime_quotes(stock)['price']) for stock in stock]
    # price_current = list(map(float,ts.get_realtime_quotes(stock)['price']))

    # pdb.set_trace()
    # high_current = float(ts.get_realtime_quotes(stock)['high'])
    time_str = time.strftime('%H:%M:%S',time.localtime())
    tmp = []
    for i in range(len(stock)):
        if traget_price[i] >0:
            if price_current[i] >traget_price[i] :
                tmp.append(time_str+ '  stock:'+stock[i]+'  已经大于目标点位！')
                print()
            else:
                print(time_str+'  stock:'+stock[i] +'  还差'+str(round(traget_price[i]-price_current[i],3))+'到达目标点位！')
        else:
            if price_current[i] < -traget_price[i]:
                tmp.append(time_str + '  stock:' + stock[i] + '  已经大于目标点位！')
                # print(tmp)
                # tmp = win32api.MessageBox(0, tmp, 'Alert',1)
                # if tmp == 1:
                #     print('Monitor,continues')
                # else:
                #     print('Monitor exits')
                #     break
            else:
                print(time_str + '  stock:' + stock[i] + '  还差' + str(round(traget_price[i] + price_current[i], 3))+'到达目标点位！')

    # '\n'.join()    使用回车将信息连接起来

    if len(tmp)>0:
        info = ''
        for i in range(len(tmp)):
            info+='\n'+tmp[i]
        tmp = win32api.MessageBox(0, info, 'Alert', 1)
        if tmp == 1:
            print('Monitor,continues')
        else:
            print('Monitor exits')
            break