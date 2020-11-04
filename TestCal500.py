# -*- coding=utf-8 -*-
import csv
import numpy as np

from sklearn.utils import shuffle

from learner.cascade import Cascade
from learner.measure import *


# 随机排列实例数，将实例划分为训练集和测试集
def shuffle_index(num_samples):
    # a = range(0, 502),502是实例数
    a = range(0, num_samples)

    # 利用shuffle函数将序列a中的元素重新随机排列
    a = shuffle(a)

    # 去实例数的一半，上取整
    length = int((num_samples + 1) / 2)
    # 上半做训练集
    train_index = a[:length]
    print("训练集下标list：", len(train_index), train_index)
    # 下半做测试集
    test_index = a[length:]
    print("测试集下标list：", len(test_index), test_index)
    return [train_index, test_index]


# 加载数据和标签
def load_csv():
    """
    从CSV文件中读取数据信息
    :param csv_file_name: CSV文件名
    :return: Data：二维数组
    """

    p = r'D:\Pycharm2020.1.3\WorkSpace\MLDF\dataset\CAL500.csv'
    with open(p, encoding='utf-8') as f:
        FullData = np.loadtxt(f, str, delimiter=",")

    label = FullData[0]
    # print("打印标签集的规模：", label.shape[0])
    data = FullData[1:]
    # print("打印数据集的规模：", data.shape[0], data.shape[1])
    print("加载CAL500数据和标签完成！！!")

    # 取数据集的行数，即是实例数
    num_samples = data.shape[0]
    # 训练集索引list，测试集索引list——每个集合251
    train_index, test_index = shuffle_index(num_samples)
    print("57行测试打印训练索引list、测试索引list：", train_index, test_index)
    # 选取初始数据集的一行作为训练数据？？？变量名有点迷
    train_data = data[train_index]

    num_labels = label.shape[0]
    train_label = label[num_labels-1]

    test_data = data[test_index]
    print("66行打印：", len(test_data), test_data)

    test_label = label

    return [train_data, train_label, test_data, test_label]


if __name__ == '__main__':
    dataset = "CAL500"
    # 初始化数据集、标签集、测试数据标签集
    train_data, train_label, test_data, test_label = load_csv()

    # 构造森林，将另个森林级联，最大层数设为10，5折交叉验证
    model = Cascade(dataset, max_layer=10, num_forests=2, n_fold=5, step=3)

    # 训练森林，传入训练集、训练标签、指标名称、每个森林中的树的数量设为40
    model.train(train_data, train_label, "hamming loss", n_estimators=40)

    test_prob = model.predict(test_data, "hamming loss")
    value = do_metric(test_prob, test_label, 0.5)
    meatures = ["hamming loss", "one-error", "coverage", "ranking loss", "average precision", "macro-auc"]
    res = zip(meatures, value)
    for item in res:
        print(item)
