# 1. 导入需要的库，模块以及数据，模型输入使用除辛烷值外的所有性质变量，输出使用辛烷值。
# 2. 建模，我们使用Xgboost模型得到辛烷值的预测输出，与目标值（辛烷值产品值）运算得到损失和精度值。
#首先要确定n_estimatorsh值，越大，模型的学习能力就会越强，模型也越容易过拟合，确定了n_estimatorsh= 60的时候效果最好，对应  确定n_estimatorsh图 。
# 3. 使用交叉验证的方法，与线性回归&随机森林回归进行对比。图二是随机森林的。线性回归的用王军的做对比。


from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from time import time
import datetime

data = loadtxt('c_norm.csv', delimiter=",") # 将一列数据转换成array数组
X = data[:,1:8+9:]
y = data[:,8]
Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.3,random_state=420)
reg = XGBR(n_estimators=60).fit(Xtrain,Ytrain)
reg.predict(Xtest) #传统接口predict
reg.score(Xtest,Ytest) #你能想出这里应该返回什么模型评估指标么？
MSE(Ytest,reg.predict(Xtest))
reg = XGBR(n_estimators=60)
CVS(reg,Xtrain,Ytrain,cv=5).mean()


#这里应该返回什么模型评估指标，还记得么？
#严谨的交叉验证与不严谨的交叉验证之间的讨论：训练集or全数据？
CVS(reg,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()
#来查看一下sklearn中所有的模型评估指标
import sklearn
sorted(sklearn.metrics.SCORERS.keys())
#使用随机森林和线性回归进行一个对比
rfr = RFR(n_estimators=60)
CVS(rfr,Xtrain,Ytrain,cv=5).mean()
CVS(rfr,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()
lr = LinearR()
CVS(lr,Xtrain,Ytrain,cv=5).mean()
CVS(lr,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()
#开启参数slient：在数据巨大，预料到算法运行会非常缓慢的时候可以使用这个参数来监控模型的训练进度
reg = XGBR(n_estimators=60,silent=True)
CVS(reg,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()



#画学习曲线
def plot_learning_curve(estimator,title, X, y,
    ax=None, #选择子图
    ylim=None, #设置纵坐标的取值范围
    cv=None, #交叉验证
    n_jobs=None #设定索要使用的线程
    ):
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y
    ,shuffle=True
    ,cv=cv
    # ,random_state=420
    ,n_jobs=n_jobs)
    if ax == None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid() #绘制网格，不是必须
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r",label="Training score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g",label="Test score")
    ax.legend(loc="best")
    return ax

cv = KFold(n_splits=5, shuffle = True, random_state=42)
plot_learning_curve(XGBR(n_estimators=60,random_state=420) ,"XGB",Xtrain,Ytrain,ax=None,cv=cv)
plt.show()
#
#判断对模型的影响
axisx = range(10,1010,50)
rs = []
for i in axisx:
    reg = XGBR(n_estimators=i,random_state=420)
    rs.append(CVS(reg,Xtrain,Ytrain,cv=cv).mean())
print(axisx[rs.index(max(rs))],max(rs))
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()
# 方差与泛化误差
axisx = range(50,1050,50)
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBR(n_estimators=i,random_state=420)
    cvresult = CVS(reg,Xtrain,Ytrain,cv=cv) #记录1-偏差
    rs.append(cvresult.mean())
#记录方差
    var.append(cvresult.var())
#计算泛化误差的可控部分
    ge.append((1 - cvresult.mean())**2+cvresult.var())
#打印R2最高所对应的参数取值，并打印这个参数下的方差
print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])
#打印方差最低时对应的参数取值，并打印这个参数下的R2
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))
#打印泛化误差可控部分的参数取值，并打印这个参数下的R2，方差以及泛化误差的可控部分
print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()

axisx = range(100,300,10)
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBR(n_estimators=i,random_state=420)
    cvresult = CVS(reg,Xtrain,Ytrain,cv=cv)
    rs.append(cvresult.mean())
    var.append(cvresult.var())
    ge.append((1 - cvresult.mean())**2+cvresult.var())

print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))
print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))
rs = np.array(rs)
var = np.array(var)*0.01
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="black",label="XGB") #添加方差线
plt.plot(axisx,rs+var,c="red",linestyle='-.')
plt.plot(axisx,rs-var,c="red",linestyle='-.')
plt.legend()
plt.show()
#看看泛化误差的可控部分如何？
plt.figure(figsize=(20,5))
plt.plot(axisx,ge,c="gray",linestyle='-.')
plt.show()

time0 = time()
print(XGBR(n_estimators=60,random_state=420).fit(Xtrain,Ytrain).score(Xtest,Ytest))
print(time()-time0)