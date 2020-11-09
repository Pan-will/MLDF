import numpy as np
from sklearn.utils import shuffle


# 随机排列实例数，将实例划分为训练集和测试集
def shuffle_index(num_samples):
    a = range(0, num_samples)
    # 利用shuffle函数将序列a中的元素重新随机排列
    a = shuffle(a)
    # 去实例数的一半，上取整
    length = int((num_samples + 1) / 2)
    # 上半做训练集
    train_index = a[:length]
    # 下半做测试集
    test_index = a[length:]
    return [train_index, test_index]

train_data_csv = r'D:\Pycharm2020.1.3\WorkSpace\MLDF\dataset\yeast_data.csv'
train_label_csv = r'D:\Pycharm2020.1.3\WorkSpace\MLDF\dataset\yeast_label.csv'

with open(train_data_csv, encoding='utf-8') as f:
    data = np.loadtxt(f, str, delimiter=",")
with open(train_label_csv, encoding='utf-8') as f:
    label = np.loadtxt(f, str, delimiter=",")

# 将数据label强制转换为指定的类型，astype函数是在副本上进行，并非修改原数组。
# 从文件中load出来的数据类型是“class 'numpy.int16'”类型，需要进行类型转化
label = label.astype("int")


print("train_data矩阵信息：", type(data[0]), data[0].shape, data.shape)
print("train_label矩阵信息：", type(label[0]), label[0].shape, label.shape)

num_samples = len(data)
print("实例数：", len(data))
train_index, test_index = shuffle_index(num_samples)
train_data = data[train_index]
train_label = label[train_index]
test_data = data[test_index]
test_label = label[test_index]
print("train_data", type(train_data), train_data.shape, len(train_data))
print("train_label", type(train_label), train_label.shape, len(train_label))
print("test_data", type(test_data), test_data.shape, len(test_data))
print("test_label", type(test_label), test_label.shape, len(test_label))





# data_csv = r'D:\Pycharm2020.1.3\WorkSpace\MLDF\dataset\CAL500_data.csv'
# label_csv = r'D:\Pycharm2020.1.3\WorkSpace\MLDF\dataset\CAL500_label.csv'
# with open(data_csv, encoding='utf-8') as f:
#     data = np.loadtxt(f, str, delimiter=",")
# with open(label_csv, encoding='utf-8') as f:
#     label = np.loadtxt(f, str, delimiter=",")
#
# print("data矩阵信息：", type(data[0]), data[0].shape, data.shape)
# label = label.astype("int")
# print("label矩阵信息：", type(label[0]), label[0].shape, label.shape)
# num_samples = len(data)
# train_index, test_index = shuffle_index(num_samples)
# train_data = data[train_index]
# train_label = label[train_index]
# test_data = data[test_index]
# test_label = label[test_index]
# print("train_data", type(train_data), train_data.shape, len(train_data))
# print("train_label", type(train_label), train_label.shape, len(train_label))
# print("test_data", type(test_data), test_data.shape, len(test_data))
# print("test_label", type(test_label), test_label.shape, len(test_label))




# dataset = "image"
# data = np.load("image_data.npy")
# print(type(data[0][0]))
# print("data矩阵信息：", data.shape, type(data[0]), data[0].shape)
# num_samples = data.shape[0]
# print("数据集的行数，即实例个数：", num_samples)
#
# trainData, testData = shuffle_index(num_samples)
# print(len(testData), len(trainData), type(trainData), trainData)
#
# train_data = data[trainData]
# test_data = data[testData]
# wangwen = data[[1, 2]]
# print(wangwen)
# print(len(wangwen), type(wangwen), wangwen.shape)
# print(len(train_data), type(train_data), train_data.shape)

# print("\n")
# label = np.load("image_label.npy")
# print(type(label[0][0]))
# print("label矩阵信息：", label.shape, type(label[0]), label[0].shape)
