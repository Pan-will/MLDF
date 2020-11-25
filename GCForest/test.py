import numpy as np
from .gcForest import *
from time import time


def load_data():
    train_data = np.load()
    train_label = np.load()
    train_weight = np.load()
    test_data = np.load()
    test_label = np.load()
    test_file = np.load()
    return [train_data, train_label, train_weight, test_data, test_label, test_file]


if __name__ == '__main__':
    train_data, train_label, train_weight, test_data, test_label, test_file = load_data()
    clf = gcForest(num_estimator=100, num_forests=4, max_layer=2, max_depth=100, n_fold=5)
    start = time()
    clf.train(train_data, train_label, train_weight)
    end = time()
    print("fitting time: " + str(end - start) + " sec")
    start = time()
    prediction = clf.predict(test_data)
    end = time()
    print("prediction time: " + str(end - start) + " sec")
    result = {}
    for index, item in enumerate(test_file):
        if item not in result:
            result[item] = prediction[index]
        else:
            result[item] = (result[item] + prediction[index]) / 2
    print(result)



# deep gcForest的伪代码：
# input = multi_Granined Scanning 的结果
# for level_i in range(num_levels):
#     # level_i层处理后的结果
#     result = level_i(input)
#     # 更新输入向量，将本层的输入和本轮的输出拼接，作为下一层的输入
#     Input = Concatenate(result, Input)
#     # 对最后一层中每个Forest的结果求均值
#     Score = AVE(最后一层的result)
#     # 将Score中值最大的最为最终预测
#     Class = MAX(Score)

