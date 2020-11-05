import numpy as np
import pandas as pd

# alist = [[[0, 1, 2, 3], [4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]]
# m = np.zeros([2, 6, 10])
# print(m[1, :])

data = pd.DataFrame(np.arange(20).reshape((5, 4)),  index=['a', 'b', 'c', 'd', 'e'])
print(data)
# print('*' * 40)
# print(data.drop(['a']))  # 删除a 行，默认inplace=False,
# print('*' * 40)
# print(data)  # data 没有变化
# print('*' * 40)
# print(data.drop(['A'], axis=1))  # 删除列
# data.drop(columns=3).values.reshape(-1, 117)
print(data.drop(columns=2, axis=1).values.reshape(-1,5))
