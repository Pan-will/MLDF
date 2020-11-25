import numpy as np

from .Layer import Layer


class KfoldWarpper:
    # 参数：森林数=2，每个森里中的树的数量=40，交叉验证的倍数=5，层序号（1~20，for循环ing），步数=3
    def __init__(self, num_forests, n_estimators, n_fold, kf, layer_index, step=3):
        self.num_forests = num_forests
        self.n_estimators = n_estimators
        self.n_fold = n_fold
        self.kf = kf
        self.layer_index = layer_index
        self.step = step
        # 最终模型集C = {layer 1，...,layer L}
        self.model = []

    def train(self, train_data, train_label):
        """
        :param train_data:训练数据
        :param train_label:对应标签
        :return:
            prob: array, whose shape is (num_samples, num_labels)，一个数组，实例数是行数，标签数是列数
            prob_concatenate
        """
        # 标签数是训练标签集的列数
        self.num_labels = train_label.shape[1]

        # 实例数、特征数分别是训练集的行数和列数，注意区分特征和标签
        num_samples, num_features = train_data.shape

        # 构造一个二维矩阵，规模是（实例数，标签数），构造的矩阵不为空
        prob = np.empty([num_samples, self.num_labels])
        # 构造一个二维矩阵，规模是（森林数，实例数，标签数）=（2，实例数，标签数），构造的矩阵不为空，用于放置预测结果
        prob_concatenate = np.empty([self.num_forests, num_samples, self.num_labels])

        fold = 0
        # train_data维度：（1000, 304）
        for train_index, test_index in self.kf:  # 进行k折交叉验证，在train_data里创建交叉验证的补充
            # train_data的shape：(1204, 294)，切片有三个参数，第一个是块下标，后面两个跟二维数组一样
            # 也就是每趟循环取一个训练数据、测试数据、训练数据对应标签
            """
            原始的数据集划成了4份：train_data（251，68）、test_data（251，68）、train_label（251，174）、test_label（251，174）；
            这里又把data部分划成了(167, 68)、(84, 68)；label部分划成了(167, 174)、(84, 174)
            X_train: <class 'numpy.ndarray'> (167, 68) 167
            X_val <class 'numpy.ndarray'> (84, 68) 84
            y_train <class 'numpy.ndarray'> (167, 174) 167
            """
            X_train = train_data[train_index, :]# 选出训练集
            X_val = train_data[test_index, :]# 验证集
            y_train = train_label[train_index, :]# 训练标签
            # weight_train = weight[train_index]  # 训练集对应的权重

            # 加入层：构建第fold个层类
            # 构建层类，参数列表：每个森林树的数量=40，森林数量=2，标签数=5（不同数据集，标签数不同），步数=3，层序号，交叉验证倍数
            layer = Layer(self.n_estimators, self.num_forests, self.num_labels, self.step, self.layer_index, fold)

            # layer层的训练，参数：训练集，对应标签
            layer.train(X_train, y_train)

            self.model.append(layer)
            fold += 1
            # 做预测，参数是新划分的test矩阵，shape是（84，68），返回值是[预测值针对森林数取得均值， 按分类器存放的预测值]
            prob[test_index], prob_concatenate[:, test_index, :] = layer.predict(X_val)
        return [prob, prob_concatenate]

    def predict(self, test_data):
        test_prob = np.zeros([test_data.shape[0], self.num_labels])
        test_prob_concatenate = np.zeros([self.num_forests, test_data.shape[0], self.num_labels])
        for layer in self.model:
            temp_prob, temp_prob_concatenate = layer.predict(test_data)
            test_prob += temp_prob
            test_prob_concatenate += temp_prob_concatenate
        test_prob /= self.n_fold
        test_prob_concatenate /= self.n_fold
        return [test_prob, test_prob_concatenate]

    def train_and_predict(self, train_data, train_label, test_data):
        prob, prob_concatenate = self.train(train_data, train_label)
        test_prob, test_prob_concatenate = self.predict(test_data)
        return [prob, prob_concatenate, test_prob, test_prob_concatenate]
