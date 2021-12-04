
from sklearn import preprocessing



train_data = [
    [1,2,3],[4,5,6],[7,8,9],
    [0,20,30],[10,2,5],[1,0,4],
    [3,5,7],[1,3,7],[3,7,0]
]
# mms = mm.fit(train_data)
# 归一化
scaler = preprocessing.MinMaxScaler().fit(train_data)
# train_label = mm.fit_transform(train_data)
out1 = scaler.transform(train_data)
print(out1)

# 反归一化
# predict_value = mm.inverse_transform(train_label)
out2 = scaler.inverse_transform(out1)
print(out2)

test = [[5,10,15]]
print(scaler.transform(test))

# from sklearn import preprocessing
# import numpy as np
# X = np.array([[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]])
# scaler= preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X)
# X_scaled = scaler.transform(X)
# print(X)
# print(X_scaled)
# X1=scaler.inverse_transform(X_scaled)
# print(X1)
# print(X1[0, -1])

