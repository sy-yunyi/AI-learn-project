import tushare as ts
import matplotlib.pyplot as plt
import pdb

stock = '600000'
# data = ts.get_k_data(stock,start= '2016-01-01',end='2017-12-30')
# price = data['close']
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(price,label = 'qfq')
# data = ts.get_k_data(stock,autype=None,start= '2016-01-01',end='2017-12-30')
# price = data['close']
# ax.plot(price, label = 'None')
# plt.legend()
# plt.show()
# 9-13



# #
# data = ts.get_k_data(stock,ktype='5',start= '2018-01-09',end='2018-01-13')
# date_list =[]
# for i in range(len(data['date'].tolist())):
#     date_list.append(data['date'].tolist()[i].split()[0])
# # print(data['date'].tolist()[0].split()[0]=='2017-11-03')
# start = date_list.index('2018-01-09')
# end = date_list.index('2018-01-15')-1
# # print(data[start:end])
# date_list_t = date_list[start:end]
# data_list = data[start:end]
#
# price_list = data_list['close']
# print(date_list_t)
# l = list(range(0,len(date_list_t)//12))
# print(l[0])
# print_date= []
# for i in range(len(l)):
#     print_date.append(price_list.tolist()[l[i]])
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# print(l)
# ax.set_xticks(l)
# ax.set_xticklabels(data[l])
# ax.plot(price_list, label = 'None')
# plt.legend()
# plt.show()


# data = ts.get_k_data(stock,ktype='5',start= '2018-01-09',end='2018-01-13')
#
# price = data['close'].values
# dates = data['date'].values
# tmp = [True if i[:10] in ['2018-01-09','2018-01-10','2018-01-11','2018-01-12','2018-01-13'] else False for i in dates]
# price =price[tmp]
# dates = dates[tmp]

