import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

class Layer:
    def __init__(self, n_estimators, num_forests, num_labels, step=3, layer_index=0, fold=0):
        """
        :param n_estimators: 森林中树的数量=40
        :param num_forests: 森林数量=2
        :param num_labels: 标签数量
        :param step: 步数=3
        :param layer_index: 层序号
        :param fold:
        """
        self.n_estimators = n_estimators
        self.num_labels = num_labels
        self.num_forests = num_forests
        self.layer_index = layer_index
        self.fold = fold
        self.step = step
        self.model = []

    def train(self, train_data, train_label):
        """
        :param train_data: 训练数据集
        :param train_label: 训练数据对应的标签
        :return:
        """
        # 在第一层中，每个森林中有40棵树，然后比上一层增加20棵树，直到树数达到100，最多100棵树；
        n_estimators = min(20 * self.layer_index + self.n_estimators, 100)
        # 最大深度 = 步数*层序号 + 步数
        max_depth = self.step * self.layer_index + self.step
        # 遍历森林块
        for forest_index in range(self.num_forests):
            # 第偶数个森林，用随机森林分类器
            if forest_index % 2 == 0:
                clf = RandomForestClassifier(n_estimators=n_estimators,
                                             criterion="gini",
                                             max_depth=max_depth,
                                             n_jobs=-1)
            # 第奇数个森林，用极端随机森林分类器
            # 一般情况下，极端随机森林分类器在分类精度和训练时间等方面都要优于随机森林分类器。
            else:
                clf = ExtraTreesClassifier(n_estimators=n_estimators,
                                           criterion="gini",
                                           max_depth=max_depth,
                                           n_jobs=-1)
            clf.fit(train_data, train_label)
            self.model.append(clf)
        self.layer_index += 1

    # 预测
    def predict(self, test_data):
        # 设置空数组，
        predict_prob = np.zeros([self.num_forests, test_data.shape[0], self.num_labels])

        for forest_index, clf in enumerate(self.model):
            predict_p = clf.predict_proba(test_data)
            for j in range(len(predict_p)):
                predict_prob[forest_index, :, j] = 1 - predict_p[j][:, 0].T

        prob_avg = np.sum(predict_prob, axis=0)
        prob_avg /= self.num_forests
        prob_concatenate = predict_prob
        return [prob_avg, prob_concatenate]

    def train_and_predict(self, train_data, train_label, val_data, test_data):
        self.train(train_data, train_label)
        val_avg, val_concatenate = self.predict(val_data)
        prob_avg, prob_concatenate = self.predict(test_data)

        return [val_avg, val_concatenate, prob_avg, prob_concatenate]

