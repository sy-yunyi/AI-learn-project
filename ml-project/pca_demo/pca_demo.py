import numpy as np
import pdb
data = np.loadtxt('wine.txt')

def pca_diy(data,n):
    mean_v = np.mean(data,axis=0)   #取得每列均值
    data_new = (data - mean_v)/np.std(data,axis=0)        # 均值归零化
    cov_mat = np.cov(data_new,rowvar = 0)
    eig_vals, eig_vects = np.linalg.eig(cov_mat)   #求得特征向量和特征值
    eig_vindex = np.argsort(eig_vals)[::-1]
    # print(eig_vals,eig_vects)
    tmp = eig_vindex[:n]

    eig_vec_selected = eig_vects[:,tmp]
    # pdb.set_trace()
    # print(eig_vals)
    # print(eig_vals[eig_vindex[0]])
    # print(sum(eig_vals))
    print(eig_vals[eig_vindex[0]]/sum(eig_vals))
    return eig_vec_selected

def pac_demo(data,n):
    mean_v = np.mean(data,axis= 0)
    data_new = data - mean_v
    cov_mat = np.cov(data_new,rowvar=0)
    eig_vals, eig_vects = np.linalg.eig(cov_mat)
    eig_vindex = np.argsort(eig_vals)[::-1]
    tmp = eig_vindex[:n]
    eig_vec_selected = eig_vects[:,tmp]
    return eig_vec_selected

vectors = pca_diy(data,2)
# data_new = np.dot(data, vectors)

# a = np.random.uniform(1,10,(5,4))
# pca_diy(a,4)
# print(a)

# print (data_new)