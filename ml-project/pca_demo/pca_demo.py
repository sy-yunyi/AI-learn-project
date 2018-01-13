import numpy as np

data = np.loadtxt('wine.txt')

def pca_diy(data,n):
    mean_v = np.mean(data,axis=0)   #取得每列均值
    data_new = data - mean_v        # 均值归零化
    cov_mat = np.cov(data_new,rowvar=0)
    eig_vals, eig_vects = np.linalg.eig(cov_mat)
    eig_vindex = np.argsort(eig_vals)[::-1]
    # print(eig_vals,eig_vects)
    tmp = eig_vindex[:n]

    eig_vec_selected = eig_vects[:,tmp]

    return eig_vec_selected
vectors = pca_diy(data,3)
data_new = np.dot(data, vectors)
print (data_new)