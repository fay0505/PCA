#原始数据信息的说明
# Attribute Information:
#1. sepal length in cm
#2. sepal width in cm
#3. petal length in cm
#4. petal width in cm
#5. class: -- Iris Setosa  -- Iris Versicolour -- Iris Virginica

import numpy as np

def load_data(path):

    #将三种类型的鸢尾花名字分别中数字代替
    name_dict = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}

    Y = [] #记录第i个样本数据的鸢尾花种类（用0 1 2 表示）

    #将文本中名字与其他的数据分开，以便用numpy处理
    with open(path, 'r') as f:
        all_lines = f.readlines()

        #存储去除掉名字的数据
        data_list = []
        for line in all_lines:
            tmp_line = line.split(',')
            tmp_list = []                   #存储每一行的数据
            for i in range(len(tmp_line)):
                if i != len(tmp_line) - 1:
                    tmp_list.append( float(tmp_line[i]))

            iris_name = tmp_line[-1].split('\n')[0]


            data_list.append(tmp_list)
            Y.append(name_dict[iris_name])

    X = np.matrix(data_list)
    #print(Y)
    #print(X, X.shape)

    return X, Y



