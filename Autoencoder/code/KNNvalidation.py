from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV  #通过网络方式来获取参数
import numpy
# 导入iris数据集
iris2=datasets.load_iris()
X2=iris2.data
y2=iris2.target
print(X2.shape,y2.shape)

# 设置需要搜索的K值，'n_neightbors'是sklearn中KNN的参数
parameters={'n_neighbors':[1,3,5,7,9,11,13,15]}
knn=KNeighborsClassifier()#注意：这里不用指定参数

# 通过GridSearchCV来搜索最好的K值。这个模块的内部其实就是对每一个K值进行评估
clf=GridSearchCV(knn,parameters,cv=5)  #5折
clf.fit(X2,y2)

# 输出最好的参数以及对应的准确率
print("最终最佳准确率：%.2f"%clf.best_score_,"最终的最佳K值",clf.best_params_)