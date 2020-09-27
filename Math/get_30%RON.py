import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import table as table
from sklearn import linear_model
from sklearn.model_selection import train_test_split


io = "./newdata_outS.csv"
io_outS = "./newdata_outS.csv"
cols2skip = [7,8]
cols = [i for i in range(43) if i not in cols2skip]
x = pd.read_csv(io,header=None,usecols=cols)
X_train = np.array(x)
X_test = np.array(pd.read_csv(io_outS,header=None,usecols=cols))
y = pd.read_csv(io,header=None,usecols=[8])
y_train = np.array(y)
y_test = np.array(pd.read_csv(io,header=None,usecols=[8]))

# X_train,X_test,y_train,y_test = train_test_split(normal_x,normal_y,test_size = 0.2,random_state=125)
regr = linear_model.LinearRegression()
clf = regr.fit(X_train,y_train)
pre = clf.predict(X_test)

RON_loss = (X_test[:,1].T-y_test.T)[0]
predic_RON_loss = (X_test[:,1].T-pre.T)[0]
print(RON_loss)
data = pd.read_csv(io)
for rownum in range(1,len(data)):
    df1 = data.columns[8]
    com_name = data.iloc[rownum][df1]
    print(com_name)
    if ((RON_loss[rownum]-predic_RON_loss[rownum])/RON_loss[rownum])>0.3:
        print("满足条件")
        data.drop(rownum)
data.to_csv("newdata2.csv")