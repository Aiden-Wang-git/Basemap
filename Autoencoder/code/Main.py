import copy

import pymysql as pymysql
from scipy.interpolate import interp1d
pymysql.install_as_MySQLdb()
import MySQLdb
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
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
from nltk.cluster.kmeans import KMeansClusterer

# 预测船舶主键ID,船舶MMSI，预测点时间
ID = "96697818"
KMeansNum = 5
MMSISelect = ""
BaseDateTimeSelect = ""
epsNum = 0.1
min_samplesNum = 10

# 读取预测船舶的信息，并将起点角度转换为-180~180，返回真实航迹real_trajectory以及第一个点的信息first_point1
def readBegin():
    db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # 查询起点
    try:
        cursor.execute("SELECT ID,MMSI,BaseDateTime,LAT,LON,SOG,COG FROM ais_2017 WHERE ID= '{0}';".format(ID))
        firstPoint = cursor.fetchall()[0]
        global MMSISelect
        MMSISelect= firstPoint[1]
        global BaseDateTimeSelect
        BaseDateTimeSelect = firstPoint[2]
    except:
        print("读取起点错误")
    # 查询真实航迹
    try:
        # 执行SQL语句,获得预测真实航迹
        cursor.execute("SELECT ID,MMSI,BaseDateTime,LAT,LON,SOG,COG FROM ais_2017 WHERE  MMSI = '{0}' order by BaseDateTime ;".format(MMSISelect))
        # 执行SQL语句,获得起点信息
        realTrajectory = list(cursor.fetchall())
    except:
        print
        "Error: unable to fecth data"
    firstPoint = list(firstPoint)
    if (firstPoint[6] < -180):
        firstPoint[6] = 360 + firstPoint[6]
    if (firstPoint[6] > 180):
        firstPoint[6] = firstPoint[6] - 360
    print("===============初始点和真实轨迹成功查询====================")
    print(firstPoint)
    return realTrajectory, firstPoint

# 根据条件获取周围相似AIS点
def getS0(firstPoint):
    # 初始点的维度、经度、速度、航向
    global MMSISelect
    first_LAT = firstPoint[3]
    first_LON = firstPoint[4]
    first_SOG = firstPoint[5]
    first_COG = firstPoint[6]
    db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # SQL 查询语句,查询起始点附近0.05°的船只信息
    sql_result = "SELECT ID,MMSI,BaseDateTime,LAT,LON,SOG,COG FROM ais_2017 WHERE lat BETWEEN {0} AND {1} AND lon BETWEEN {2} AND {3} AND MMSI!='{4}';".format(
        first_LAT - 0.03, first_LAT + 0.03, first_LON - 0.03, first_LON + 0.03, MMSISelect)
    try:
        # 执行SQL语句,获得预测真实航迹
        cursor.execute(sql_result)
        result_trajectory = list(cursor.fetchall())
    except:
        print
        "查询周围AIS点错误"
    resultTrajectory = []
    for i in range(len(result_trajectory)):
        result_trajectory[i] = list(result_trajectory[i])
        if (result_trajectory[i][6] < -180):
            result_trajectory[i][6] = 360 + result_trajectory[i][6]
        if (result_trajectory[i][6] > 180):
            result_trajectory[i][6] = result_trajectory[i][6] - 360
        if abs(first_COG - result_trajectory[i][6]) < 45 or abs(first_COG - result_trajectory[i][6]) > 315:
            if abs(first_SOG - result_trajectory[i][5]) < 5: #航速差小于5节
                resultTrajectory.append(result_trajectory[i][1:])
    print("=================周围找初始类完成===========================")
    S0 = []
    for i in range(len(resultTrajectory)):
        if getDistance([first_LAT,first_LON],resultTrajectory[i][2:4]) < 1:
            S0.append(resultTrajectory[i])
    temp = {}
    for MMSI, BaseDateTime, LAT, LON, SOG, COG in S0:
        if MMSI not in temp:  # we see this key for the first time
            temp[MMSI] = (MMSI, BaseDateTime, LAT, LON, SOG, COG)
        else:
            # 找出相同MMSI中距离预测点最近的那个点
            if getDistance([first_LAT,first_LON], [ LAT, LON]) < getDistance([first_LAT,first_LON],temp[MMSI][2:4]):
                temp[MMSI] = (MMSI, BaseDateTime, LAT, LON, SOG, COG)
    S0 = list(temp.values())
    print("============得到初始类S0=============")
    return S0

# 提取S0以及firstPoint前置、后置轨迹,时间间隔为30S，前后各30min
def getTrajectory(S0,firstPoint):
    del firstPoint[0] # 将起始点的数据ID删除
    backTrajectory = []
    forwardTrajectory = []
    print("SO中一共会有", len(S0), "条航迹")
    S0.append(firstPoint)
    for i in range(len(S0)):
        now_time = S0[i][1]
        begin_time_date = now_time - datetime.timedelta(hours=6)
        begin_time = datetime.datetime.strftime(begin_time_date, '%Y-%m-%d %H:%M:%S')
        end_time_date = now_time + datetime.timedelta(hours=6)
        end_time = datetime.datetime.strftime(end_time_date, '%Y-%m-%d %H:%M:%S')
        db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8')
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # SQL 查询语句
        sql_S0 = "SELECT ID,MMSI,BaseDateTime,LAT,LON,SOG,COG FROM ais_2017 WHERE  MMSI = '{0}' and  lat BETWEEN {1} AND {2} AND lon BETWEEN {3} AND {4} and BaseDateTime between '{5}' and '{6}' order by BaseDateTime;".format(
            S0[i][0], S0[i][2] - 2, S0[i][2] + 2, S0[i][3] - 2, S0[i][3] + 2, begin_time, end_time)
        try:
            # 执行SQL语句,获得周围目标船舶的航迹
            cursor.execute(sql_S0)
            S0_tajectory = cursor.fetchall()
        except:
            print
            "从数据库中轨迹提取出现错误"
        first_time = S0[i][1]
        date = [l[2] for l in S0_tajectory]
        # 将时间转换为时间戳
        dateStamp = []
        for k in range(len(date)):
            dateStamp.append(int(date[k].timestamp()))
        LAT = [k[3] for k in S0_tajectory]
        LON = [k[4] for k in S0_tajectory]
        SOG = [k[5] for k in S0_tajectory]
        COG = [k[6] for k in S0_tajectory]
        for k in range(len(COG)):
            if (COG[k] < -180):
                COG[k] = 360 + COG[k]
            if (COG[k] > 180):
                COG[k] = COG[k] - 360
        X = pd.date_range(start=first_time - datetime.timedelta(minutes=30), periods=121, freq='30S')
        X = list(X)
        XStamp = []
        for h in range(len(X)):
            XStamp.append(int(X[h].to_pydatetime().timestamp()))
        if (XStamp[0] - dateStamp[0] < 1800) or (dateStamp[len(dateStamp) - 1] - XStamp[120] < 1800):
            print('轨迹', i, '太长or太短，无法插值')
            continue
        f_LAT = interpolate.interp1d(dateStamp, LAT, kind="linear")
        f_LON = interpolate.interp1d(dateStamp, LON, kind="linear")
        f_SOG = interpolate.interp1d(dateStamp, SOG, kind="linear")
        f_COG = interpolate.interp1d(dateStamp, COG, kind="linear")
        LATS = f_LAT(XStamp)
        LONS = f_LON(XStamp)
        SOGS = f_SOG(XStamp)
        COGS = f_COG(XStamp)
        back = []
        forward = []
        for j in range(0, 60):
            back.append([S0[i][0], X[j], LATS[j], LONS[j], SOGS[j], COGS[j]])
        backTrajectory.append(back)
        for m in range(61, 121):
            forward.append([S0[i][0], X[m], LATS[m], LONS[m], SOGS[m], COGS[m]])
        forwardTrajectory.append(forward)
        print("轨迹", i, "完成！")
    backSelect = backTrajectory[len(backTrajectory) - 1]
    forwardSelect = forwardTrajectory[len(backTrajectory) - 1]
    del backTrajectory[len(backTrajectory) - 1]
    del forwardTrajectory[len(forwardTrajectory) - 1]
    print("================轨迹提取完成=================")
    return backTrajectory, forwardTrajectory, backSelect, forwardSelect
    
# 依据backTrajectory+forwardTrajectory对轨迹进行聚类
def clusterKMeans(backTrajectory,forwardTrajectory):
    trajectory = []
    trajectorySingle = []
    trajectoryChain = []  # 得到60*4=240维的数据
    for i in range(len(backTrajectory)):
        trajectory.append(backTrajectory[i]+forwardTrajectory[i])
    # for i in range(0,len(trajectory)):
    #     for j in range(0,len(trajectory[i])):
    #         trajectorySingle.extend(trajectory[i][j][2:])
    #     trajectoryChain.append(trajectorySingle)

    label = DBSCAN(min_samples=2, metric=getDistanceTrajectory, eps=3,p=1).fit(np.array(trajectory))
    print("KMeans分类任务完成")

# ==================工具函数===============
# 计算A、B两点实际距离
def getDistance(pointA=[], pointB=[]):
    distance = geodesic((pointA[0], pointA[1]), (pointB[0], pointB[1])).nm
    return distance

# 计算两条轨迹之间的距离
def getDistanceTrajectory(trajectoryA = [],trajectoryB = []):
    distance = 0.0
    for i in range(len(trajectoryA)):
        distance+= getDistance(trajectoryA[i][2:4],trajectoryB[i][2:4])
    return distance

def out():
    realTrajectory, firstPoint = readBegin()
    S0 = getS0(firstPoint)
    backTrajectory, forwardTrajectory, backSelect, forwardSelect = getTrajectory(S0,firstPoint)
    backTrajectoryLabel,forwardTrajectoryLabel = clusterKMeans(backTrajectory,forwardTrajectory)

out()