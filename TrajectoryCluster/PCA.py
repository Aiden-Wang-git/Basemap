import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing

# 数据导入
csv_file = "data.csv"
csv_data = pd.read_csv(csv_file, low_memory=False)  # 防止弹出警告
csv_df = pd.DataFrame(csv_data)

# 数据标准化
scaler = preprocessing.MinMaxScaler().fit(csv_df)
X_scaler = pd.DataFrame(scaler.transform(csv_df))


# 主成分分析建模
pca = PCA(n_components=None)  # n_components提取因子数量
# n_components=‘mle’，将自动选取主成分个数n，使得满足所要求的方差百分比
# n_components=None，返回所有主成分
pca.fit(X_scaler)
pca.explained_variance_  # 贡献方差，即特征根
pca.explained_variance_ratio_  # 方差贡献率
pca.components_  # 成分矩阵
k1_spss = pca.components_ / np.sqrt(pca.explained_variance_.reshape(-1, 1))  # 成分得分系数矩阵

# 确定权重
# 求指标在不同主成分线性组合中的系数
j = 0
Weights = []
for j in range(len(k1_spss)):
    for i in range(len(pca.explained_variance_)):
        Weights_coefficient = np.sum(100 * (pca.explained_variance_ratio_[i]) * (k1_spss[i][j])) / np.sum(
            pca.explained_variance_ratio_)
    j = j + 1
    Weights.append(np.float(Weights_coefficient))
print('Weights',Weights)


Weights=pd.DataFrame(Weights)
Weights1 = preprocessing.MinMaxScaler().fit(Weights)
Weights2 = Weights1.transform(Weights)
print('Weights2',Weights2)

