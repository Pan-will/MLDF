from sklearn.ensemble import ExtraTreesRegressor  # 引入极端森林回归
from sklearn.ensemble import RandomForestRegressor  # 引入随机森林回归
import numpy as np

# 定义层类
class Layer:
    def __init__(self, n_estimators, num_forests, max_depth=30, min_samples_leaf=1):
        self.num_forests = num_forests  # 定义森林数
        self.n_estimators = n_estimators  # 每个森林的树个数
        self.max_depth = max_depth  # 每一颗树的最大深度
        self.min_samples_leaf = min_samples_leaf  # 树会生长到所有叶子都分到一个类，或者某节点所代表的样本数已小于min_samples_leaf
        self.model = []  # 最后产生的类向量

    def train(self, train_data, train_label, weight, val_data):  # 训练函数
        val_prob = np.zeros([self.num_forests, val_data.shape[
            0]])  # 定义出该层的类向量，有self.num_forersts行，val_data.shape[0]列，这里我们认为val_data应该就是我们的weight

        for forest_index in range(self.num_forests):  # 对具体的layer内的森林进行构建
            if forest_index % 2 == 0:  # 如果是第偶数个，设为随机森林
                clf = RandomForestRegressor(n_estimators=self.n_estimators,  # 子树的个数,
                                            n_jobs=-1,  # cpu并行树，-1表示和cpu的核数相同
                                            max_depth=self.max_depth,  # 最大深度
                                            min_samples_leaf=self.min_samples_leaf)
                clf.fit(train_data, train_label, weight)  # weight是取样比重Sample weights
                val_prob[forest_index, :] = clf.predict(val_data)  # 记录类向量
            else:  # 如果是第奇数个，就设为极端森林
                clf = ExtraTreesRegressor(n_estimators=self.n_estimators,  # 森林所含树的个数
                                          n_jobs=-1,  # 并行数
                                          max_depth=self.max_depth,  # 最大深度
                                          min_samples_leaf=self.min_samples_leaf)  # 最小叶子限制
                clf.fit(train_data, train_label, weight)
                val_prob[forest_index, :] = clf.predict(val_data)  # 记录类向量

            self.model.append(clf)  # 组建layer层

        val_avg = np.sum(val_prob, axis=0)  # 按列进行求和
        val_avg /= self.num_forests  # 求平均
        val_concatenate = val_prob.transpose((1, 0))  # 对记录的类向量矩阵进行转置
        return [val_avg, val_concatenate]  # 返回平均结果和转置后的类向量矩阵

    def predict(self, test_data):  # 定义预测函数，也是最后一层的功能
        predict_prob = np.zeros([self.num_forests, test_data.shape[0]])
        for forest_index, clf in enumerate(self.model):
            predict_prob[forest_index, :] = clf.predict(test_data)

        predict_avg = np.sum(predict_prob, axis=0)
        predict_avg /= self.num_forests
        predict_concatenate = predict_prob.transpose((1, 0))
        return [predict_avg, predict_concatenate]


class KfoldWarpper:  # 定义每个树进行训练的所用的数据
    def __init__(self, num_forests, n_estimators, n_fold, kf, layer_index, max_depth=31,
                 min_samples_leaf=1):  # 包括森林树，森林使用树的个数，k折的个数，k-折交叉验证，第几层，最大深度，最小叶子节点限制
        self.num_forests = num_forests
        self.n_estimators = n_estimators
        self.n_fold = n_fold
        self.kf = kf
        self.layer_index = layer_index
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model = []

    def train(self, train_data, train_label, weight):
        num_samples, num_features = train_data.shape

        val_prob = np.empty([num_samples])
        # 创建新的空矩阵，num_samples行，num_forest列，用于放置预测结果
        val_prob_concatenate = np.empty([num_samples, self.num_forests])

        for train_index, test_index in self.kf:  # 进行k折交叉验证，在train_data里创建交叉验证的补充
            X_train = train_data[train_index, :]  # 选出训练集
            X_val = train_data[test_index, :]  # 验证集
            y_train = train_label[train_index]  # 训练标签
            weight_train = weight[train_index]  # 训练集对应的权重

            # 加入层
            layer = Layer(self.n_estimators, self.num_forests, self.max_depth, self.min_samples_leaf)
            # 记录输出的结果
            val_prob[test_index], val_prob_concatenate[test_index, :] = layer.train(X_train, y_train, weight_train, X_val)
            self.model.append(layer)  # 在模型中填充层级，这也是导致程序吃资源的部分，每次进行
        return [val_prob, val_prob_concatenate]

    def predict(self, test_data):  # 定义预测函数，用做下一层的训练数据

        test_prob = np.zeros([test_data.shape[0]])
        test_prob_concatenate = np.zeros([test_data.shape[0], self.num_forests])
        for layer in self.model:
            temp_prob, temp_prob_concatenate = \
                layer.predict(test_data)

            test_prob += temp_prob
            test_prob_concatenate += temp_prob_concatenate
        test_prob /= self.n_fold
        test_prob_concatenate /= self.n_fold

        return [test_prob, test_prob_concatenate]
