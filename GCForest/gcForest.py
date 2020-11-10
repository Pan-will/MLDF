from sklearn.model_selection import KFold
from .layer import *
import numpy as np


def compute_loss(target, predict):  # 对数误差函数
    temp = np.log(abs(target + 1)) - np.log(abs(predict + 1))
    res = np.dot(temp, temp) / len(temp)  # 向量点成后平均
    return res


class gcForest:  # 定义gcforest模型
    def __init__(self, num_estimator, num_forests, max_layer=2, max_depth=31, n_fold=5):
        self.num_estimator = num_estimator
        self.num_forests = num_forests
        self.n_fold = n_fold
        self.max_depth = max_depth
        self.max_layer = max_layer
        self.model = []

    def train(self, train_data, train_label, weight):
        num_samples, num_features = train_data.shape

        # basis process
        train_data_new = train_data.copy()

        # return value
        val_p = []
        best_train_loss = 0.0
        layer_index = 0
        best_layer_index = 0
        bad = 0

        kf = KFold(2, True, self.n_fold).split(train_data_new.shape[0])
        # 这里加入k折交叉验证
        while layer_index < self.max_layer:

            print("layer " + str(layer_index))
            layer = KfoldWarpper(self.num_forests, self.num_estimator, self.n_fold, kf, layer_index, self.max_depth,
                                 1)  # 其实这一个layer是个夹心layer，是2层layer的平均结果

            val_prob, val_stack = layer.train(train_data_new, train_label, weight)
            # 使用该层进行训练
            train_data_new = np.concatenate([train_data, val_stack], axis=1)
            # 将该层的训练结果也加入到train_data中
            temp_val_loss = compute_loss(train_label, val_prob)
            print("val   loss:" + str(temp_val_loss))

            if best_train_loss < temp_val_loss:  # 用于控制加入的层数，如果加入的层数较多，且误差没有下降也停止运行
                bad += 1
            else:
                bad = 0
                best_train_loss = temp_val_loss
                best_layer_index = layer_index
            if bad >= 3:
                break

            layer_index = layer_index + 1

            self.model.append(layer)

        for index in range(len(self.model), best_layer_index + 1, -1):  # 删除多余的layer
            self.model.pop()

    def predict(self, test_data):
        test_data_new = test_data.copy()
        test_prob = []
        for layer in self.model:
            predict, test_stack = layer.predict(test_data_new)
            test_data_new = np.concatenate([test_data, test_stack], axis=1)
        return predict
