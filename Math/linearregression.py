import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#读取数据
io = "./newdata.csv"
cols2skip = [7,8]
cols = [i for i in range(43) if i not in cols2skip]
x = pd.read_csv(io,header=None,usecols=cols)
normal_x = np.array(x)
y = pd.read_csv(io,header=None,usecols=[8])
normal_y = np.array(y)

X_train,X_test,y_train,y_test = train_test_split(normal_x,normal_y,test_size = 0.2,random_state=125)
regr = linear_model.LinearRegression()
clf = regr.fit(X_train,y_train)
pre = clf.predict(X_test)
print("预测辛烷的score:",regr.score(X_test,y_test))
#预测之后的辛烷损失值
pre_RONloss = X_test[:,1]-pre.T
#真实辛烷损失值
RONloss = X_test[:,1]-y_test.T
plt.rcParams['font.sans-serif']=['Simhei']
plt.plot(pre_RONloss.T,color = 'r',label='预测RON损失') #预测的
plt.plot(RONloss.T,color='g',label='真实RON损失')#真实的
plt.ylabel('RON损失')
plt.xlabel('样本')
plt.legend()
plt.show()
plt.plot(X_test[:,1],color='g',label='RON真实')
plt.plot(pre,color='r',label='RON预测')
plt.legend()
plt.show()
array = pre-y_test
# 求均值
arr_mean = np.mean(array)
print('RON损失预测误差均值：',arr_mean)
# 求方差
arr_var = np.var(array)
print('RON损失预测误差方差：',arr_var)
# 求标准差
arr_std = np.std(array, ddof=1)
print('RON损失预测误差标准差：',arr_std)