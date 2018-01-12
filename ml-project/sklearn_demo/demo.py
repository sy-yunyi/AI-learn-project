from sklearn import neighbors

import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.normal(50,6,200)
y1 = np.random.normal(5,0.5,200)

x2 = np.random.normal(30,6,200)
y2 = np.random.normal(4,0.5,200)

x3 = np.random.normal(45,6,200)
y3 = np.random.normal(2.5,0.5,200)

plt.scatter(x1,y1,c='b',marker='s',s=50,alpha=0.8)
plt.scatter(x2,y2,c='r',marker='<',s=50,alpha=0.8)
plt.scatter(x3,y3,c='g',s=50,alpha=0.8)
# plt.show()

x_val = np.concatenate((x1,x2,x3))
y_val = np.concatenate((y1,y2,y3))

#归一化
x_diff = max(x_val) - min(x_val)
y_diff = max(y_val) - min(y_val)

x_norm = [x / x_diff for x in x_val]
y_norm = [y / y_diff for y in y_val]
xy_norm =list(zip(x_norm,y_norm))
#生成分类标签数据
labels = [1] * 200 + [2] * 200 + [3] *200

#生成sk-learn的最近k近邻分类功能，参数中，n_neighbors设为30，其他为默认值

clf = neighbors.KNeighborsClassifier(30)
# clf = neighbors.KNeighborsClassifier(algorithm= 'kd_tree',leaf_size=30,p=30)
clf.fit(xy_norm,labels)

#获取距离目标最近的几个点
in_x_1 = (50 / x_diff,5 / y_diff)
in_x_2 = (30 / x_diff,3 / y_diff)
nearest = clf.kneighbors([in_x_1,in_x_2],5,False)
print(nearest)

#判断类别
prediction = clf.predict([in_x_1,in_x_2])
print(prediction)

#判定目标类别概率
prediction_proba = clf.predict_proba([in_x_1,in_x_2])
print(prediction_proba)

#测试分类模型准确率

x1_test = np.random.normal(50,6,200)
y1_test = np.random.normal(5,0.5,200)

x2_test = np.random.normal(30,6,200)
y2_test = np.random.normal(4,0.5,200)

x3_test = np.random.normal(45,6,200)
y3_test = np.random.normal(2.5,0.5,200)

xy_test_norm = zip(np.concatenate((x1_test,x2_test,x3_test)) / x_diff,np.concatenate((y1_test,y2_test,y3_test)) / y_diff)
xy_test_norm = list(xy_test_norm)
labels_test = [1] * 200 +[2] *200 + [3] *200

#生成完测试数据，进行测试得分
score = clf.score(xy_test_norm,labels)
print(score)

#使用1NN分类，会出现过拟合现象，导致准确率降低

clf_1 = neighbors.KNeighborsClassifier(1)
clf_1.fit(xy_norm,labels)
score_1 = clf_1.score(xy_test_norm,labels_test)
print(score_1)
