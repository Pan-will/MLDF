
from sklearn.utils import shuffle

from learner.cascade import Cascade
from learner.measure import *


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


# 加载数据和标签
def make_data(dataset):
    data = np.load("dataset/{}_data.npy".format(dataset))
    label = np.load("dataset/{}_label.npy".format(dataset))
    # astype转换数据类型：将标签转成int类型
    label = label.astype("int")
    # 取数据集的行数，即是实例数
    num_samples = data.shape[0]
    # 训练集索引，测试集索引
    train_index, test_index = shuffle_index(num_samples)
    train_data = data[train_index]
    train_label = label[train_index]
    test_data = data[test_index]
    test_label = label[test_index]
    return [train_data, train_label, test_data, test_label]


if __name__ == '__main__':
    dataset = "image"
    # 初始化数据集、标签集、测试数据标签集
    train_data, train_label, test_data, test_label = make_data(dataset)
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
