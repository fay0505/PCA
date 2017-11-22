#对所有样本进行中心化

import numpy as np

def  pca(X, K):


    mean_data = np.mean(X, axis=0)  #对矩阵进行纵向求平均值


    X = X - mean_data    #所有样本进行中心化


    Cov_mat = np.cov(X.T)   #计算样本的协方差矩阵

    eigen_values, eigen_vectors = np.linalg.eig(Cov_mat)   #计算协方差矩阵的特征值和特征向量

    print('eigenvalues:', eigen_values)
    print('eigenvectors:\n', eigen_vectors)
    evalues = np.argsort(-eigen_values)  #将特征值进行从大到小的排序，默认为从小到大，故加负号
    topK_evalues = evalues[0:K]    #选出最大的前K个特征值的下标


    select_vectors = eigen_vectors.T[topK_evalues]    #选取对应的特征向量

   # print('select_vector:',select_vectors,select_vectors.shape)



    finaldata = X * select_vectors.T     #表示降维后的数据
    recondata = (finaldata * select_vectors) + mean_data  #进行了坐标变化之后的数据

    #print('final:',finaldata.shape)
    #print('recondata:', recondata.shape)

    return finaldata
