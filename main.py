from LoadData import load_data
from  Pca import pca
import  numpy as np
import  matplotlib.pyplot as plt
import matplotlib.lines as mlines

if __name__ == '__main__':

    Path = 'iris.data'
    X, Y = load_data(Path)


    K = 2        #将数据降维为2维
    finaldata = pca(X, K)

    #分别存储不同类别的鸢尾花降维后的数据
    red_x = []
    red_y = []

    blue_x = []
    blue_y = []

    green_x = []
    green_y = []

    finaldata = np.array(finaldata)  #将矩阵转化为数组，便于索引

    for i in range(0,finaldata.shape[0]):    #按行遍历
        if Y[i] == 0:
            red_x.append(finaldata[i][0])
            red_y.append(finaldata[i][1])
        elif Y[i] == 1:
            blue_x.append(finaldata[i][0])
            blue_y.append(finaldata[i][1])
        else:
            green_x.append(finaldata[i][0])
            green_y.append(finaldata[i][1])


    plt.scatter(red_x, red_y, c = 'r', marker='x',label = 'Iris Setosa')
    plt.scatter(blue_x,blue_y, c = 'b', marker='*', label = 'Iris Versicolour')
    plt.scatter(green_x,green_y, c = 'g', marker='d', label = 'Iris-virginica')
    plt.legend(loc = 'upper right')
    plt.savefig('result.png', dpi = 600)
    plt.show()


