import copy
import numpy
import matplotlib.pyplot as plt
import scaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import  pandas as pd
import  os
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import pymysql as pymysql


pymysql.install_as_MySQLdb()
import MySQLdb
from math import radians, cos, sin, asin, sqrt
import numpy
import numpy as np
from geopy.distance import geodesic
import time, datetime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import interpolate
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from itertools import chain
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import math

# 预测船舶主键ID
ID = "1900"

# 获取select_point
def readData():
    db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # SQL 查询语句
    sql = "SELECT * FROM ais_2018_01_02 WHERE  ID = '{0}' order by BaseDateTime ;".format(ID)
    try:
        # 执行SQL语句,获得预测真实航迹
        cursor.execute(sql)
        select_point = cursor.fetchall()
    except:
        print
        "Error: unable to fecth data"
    select_point = [select_point[0][1],select_point[0][2],select_point[0][3],select_point[0][4],select_point[0][5],select_point[0][6]]
    return select_point

# 插值，得到新的数据，时间间隔为30S
def getTrajectory(select_point):
    db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # SQL 查询语句
    sql = "SELECT MMSI,BaseDateTime,LAT,LON,SOG,COG FROM ais_2018_01_02 WHERE  MMSI = '{0}' order by BaseDateTime ;".format(select_point[0])
    try:
        # 执行SQL语句,获得预测真实航迹
        cursor.execute(sql)
        select_trajectory_row = cursor.fetchall()
    except:
        print
        "Error: unable to fecth data"
    select_trajectory = []

    # 采用时间间隔为60S
    date = [l[1] for l in select_trajectory_row]
    # 将时间转换为时间戳
    dateStamp = []
    for k in range(len(date)):
        dateStamp.append(int(date[k].timestamp()))
    LAT = [k[2] for k in select_trajectory_row]
    LON = [k[3] for k in select_trajectory_row]
    SOG = [k[4] for k in select_trajectory_row]
    COG = [k[5] for k in select_trajectory_row]
    print("开始时间：",select_trajectory_row[0][1])
    print("结束时间：",select_trajectory_row[len(select_trajectory_row)-1][1])
    begin_time = int((select_trajectory_row[0][1]).timestamp())
    end_time = int((select_trajectory_row[len(select_trajectory_row)-1][1]).timestamp())
    X = pd.date_range(start=select_trajectory_row[0][1],end=select_trajectory_row[len(select_trajectory_row)-1][1],  freq='60S')
    X = list(X)
    XStamp = []
    for h in range(len(X)):
        XStamp.append(int(X[h].to_pydatetime().timestamp()))

    # 插值
    f_LAT = interpolate.interp1d(dateStamp,LAT,kind="linear")
    f_LON = interpolate.interp1d(dateStamp,LON,kind="linear")
    f_SOG = interpolate.interp1d(dateStamp,SOG,kind="linear")
    f_COG = interpolate.interp1d(dateStamp,COG,kind="linear")
    LATS = f_LAT(XStamp)
    LONS = f_LON(XStamp)
    SOGS = f_SOG(XStamp)
    COGS = f_COG(XStamp)
    for j in range(len(LATS)):
        timeArray = time.localtime(XStamp[j])
        otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        select_trajectory.append([otherStyleTime,LATS[j],LONS[j],SOGS[j],COGS[j]])
    return select_trajectory

# RNN中的LSTM预测
def RNN_LSTM(select_trajectory):
    select_trajectory = select_trajectory[1:]
    # 70%作为训练集，30%作为测试集
    train_size = int(len(select_trajectory) * 0.7)
    trainlist = select_trajectory[:train_size]
    testlist = select_trajectory[train_size:]
    look_back = 3
    trainX, trainY = create_dataset(trainlist, look_back)
    testX, testY = create_dataset(testlist, look_back)
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 4))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 4))
    #训练模型
    model = Sequential()
    model.add(LSTM(4, input_shape=(None, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    #做预测
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # 反归一化
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)
    plt.plot(trainY, color='green')  # 绿色是真实数据
    plt.plot(trainPredict[1:], color='red')  # 红色是预测数据
    plt.title('train_set')
    plt.show()
    plt.savefig('训练.png')
    plt.plot(testY, color='green')
    plt.plot(testPredict[1:], color='red')
    plt.title('test_set')
    plt.savefig('测试.png')
    plt.show()


#这里的look_back与timestep相同，即利用前面多少步预测下一步
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)


select_point =  readData()
select_trajectory = getTrajectory(select_point)
RNN_LSTM(select_trajectory)