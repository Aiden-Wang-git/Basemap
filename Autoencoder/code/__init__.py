import copy

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

# 预测船舶ID,预测点时间
MMSI = "367390380"
Basedatetime = "2017-02-02 06:47:54"


# 读取预测船舶的信息，返回真实航迹real_trajectory以及第一个点的信息first_point1
def readdata():
    db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # SQL 查询语句
    sql_truth = "SELECT * FROM aispoints WHERE  MMSI = '{0}' order by BaseDateTime ;".format(MMSI)
    sql_truth_point = "SELECT * FROM aispoints WHERE  MMSI = '{0}'  and BaseDateTime = '{1}';".format(MMSI,Basedatetime)
    try:
        # 执行SQL语句,获得预测真实航迹
        cursor.execute(sql_truth)
        real_tajectory1 = cursor.fetchall()
        # 执行SQL语句,获得起点信息
        cursor.execute(sql_truth_point)
        first_point1 = cursor.fetchall()
    except:
        print
        "Error: unable to fecth data"
    return real_tajectory1, first_point1


# 坐标的提取以及坐标系的转换，返回起始点first_point以及周围点result_trajectory
def zuobiaozhou(first_point2=[]):
    # 初始点的维度、经度、速度、航向
    print(first_point2)
    first_point1 = list(first_point2)
    first_LAT = first_point1[3]
    first_LON = first_point1[4]
    first_SOG = first_point1[5]
    first_COG = first_point1[6]
    db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # SQL 查询语句,查询起始点附近0.05°的船只信息
    sql_result = "SELECT * FROM aispoints WHERE lat BETWEEN {0} AND {1} AND lon BETWEEN {2} AND {3} AND MMSI!='{4}';".format(
        first_LAT - 0.05, first_LAT + 0.05, first_LON - 0.05, first_LON + 0.05, MMSI)
    try:
        # 执行SQL语句,获得预测真实航迹
        cursor.execute(sql_result)
        result_trajectory2 = cursor.fetchall()
    except:
        print
        "Error: unable to fecth data"
    # # 将起始点坐和其余数据点标转换
    # R =[[cos(first_COG),-sin(first_COG)],[sin(first_COG),cos(first_COG)]]
    # first_point1[4],first_point1[3] = np.dot(R,np.transpose([first_LON,first_LAT]))
    # for i in range(len(result_trajectory1)):
    #     result_trajectory1[i]=list(result_trajectory1[i])
    #     result_trajectory1[i][4],result_trajectory1[i][3]=np.dot(R,np.transpose([result_trajectory1[i][4],result_trajectory1[i][3]]))
    # 比较距离以及航向的
    result_trajectory1 = list(result_trajectory2)
    result_trajectory = [[]]
    first_point = [first_point1[1], first_point1[2], first_point1[3], first_point1[4], first_point1[5], first_point1[6]]
    for i in range(len(result_trajectory1)):
        if abs(first_point[5] - result_trajectory1[i][6]) < 15 or abs(first_point[5] - result_trajectory1[i][6]) > 345:
            if abs(first_point[4] - result_trajectory1[i][5]) < 2:
                result_trajectory.append([result_trajectory1[i][1], result_trajectory1[i][2], result_trajectory1[i][3],
                                          result_trajectory1[i][4], result_trajectory1[i][5], result_trajectory1[i][6]])
    del (result_trajectory[0])
    return first_point, result_trajectory


# 根据经纬度确定两点之间的距离
def getDistanceDiffrentLat(Lat1, Lat2, Lon):
    distance = geodesic((Lat1, Lon), (Lat2, Lon)).nm
    return distance


def getDistanceDiffrentLon(Lon1, Lon2, Lat):
    distance = geodesic((Lat, Lon1), (Lat, Lon2)).nm
    return distance


def getDistance(pointA=[], pointB=[]):
    distance = geodesic((pointA[2], pointA[3]), (pointB[2], pointB[3])).nm
    return distance


# 输入预测点和周围数据点，返回初始类S0
def getS0(first_point, result_trajectory):
    S0 = [[]]
    del (S0[0])
    for i in range(len(result_trajectory)):
        if getDistance(first_point, result_trajectory[i]) < 2:
            S0.append(result_trajectory[i])
    temp = {}
    for MMSI, BaseDateTime, LAT, LON, SOG, COG in S0:
        if MMSI not in temp:  # we see this key for the first time
            temp[MMSI] = (MMSI, BaseDateTime, LAT, LON, SOG, COG)
        else:
            # 找出相同MMSI中距离预测点最近的那个点
            if getDistance(first_point, [MMSI, BaseDateTime, LAT, LON, SOG, COG]) < getDistance(first_point,temp[MMSI]):
                temp[MMSI] = (MMSI, BaseDateTime, LAT, LON, SOG, COG)
    S0 = temp.values()
    S0 = list(S0)
    return S0


# 轨迹提取，根据S0，分别向后和向前提取60个点，时间间隔为30S，向前向后都是30min,返回back_trajectory和forward_trajectory,back_select即预测船只的已走轨迹
def getTrajectory(S0, first_point):
    back_trajectory = []
    forward_trajectory = []
    print("SO中一共会有", len(S0), "条航迹")
    S0.append(first_point)
    for i in range(len(S0)):
        db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8')
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # SQL 查询语句
        sql_S0 = "SELECT * FROM aispoints WHERE  MMSI = '{0}' and  lat BETWEEN {1} AND {2} AND lon BETWEEN {3} AND {4} order by BaseDateTime;".format(
            S0[i][0], S0[i][2] - 2, S0[i][2] + 2, S0[i][3] - 2, S0[i][3] + 2)
        try:
            # 执行SQL语句,获得周围目标船舶的航迹
            cursor.execute(sql_S0)
            S0_tajectory = cursor.fetchall()
        except:
            print
            "Error: unable to fecth data"
        date = []
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
        X = pd.date_range(start=first_time - datetime.timedelta(minutes=60), periods=121, freq='60S')
        X = list(X)
        XStamp = []
        for h in range(len(X)):
            XStamp.append(int(X[h].timestamp()))
        f_LAT = interpolate.interp1d(dateStamp, LAT, kind="quadratic")
        f_LON = interpolate.interp1d(dateStamp, LON, kind="quadratic")
        f_SOG = interpolate.interp1d(dateStamp, SOG, kind="quadratic")
        f_COG = interpolate.interp1d(dateStamp, COG, kind="quadratic")
        LATS = f_LAT(XStamp)
        LONS = f_LON(XStamp)
        SOGS = f_SOG(XStamp)
        COGS = f_COG(XStamp)
        back = []
        forward = []
        for j in range(0, 60):
            back.append([S0[i][0], X[j], LATS[j], LONS[j], SOGS[j], COGS[j]])
        back_trajectory.append(back)
        for m in range(61, 121):
            forward.append([S0[i][0], X[m], LATS[m], LONS[m], SOGS[m], COGS[m]])
        forward_trajectory.append(forward)
        print("轨迹", i, "完成！")
    back_select = back_trajectory[len(back_trajectory) - 1]
    forward_select = forward_trajectory[len(back_trajectory) - 1]
    del back_trajectory[len(back_trajectory) - 1]
    del forward_trajectory[len(forward_trajectory) - 1]
    print("================轨迹提取完成=================")
    return back_trajectory, forward_trajectory, back_select,forward_select

# PCA降维,输入forward_trajectory，返回降维之后的PCA_forward=[[LAT,LON,SOG,COG]]，降至3维。
def PCA_3(forward_trajectory):
    pca_sk = PCA(n_components=3)
    forward_trajectory_chain = []
    MMSI = []
    for single_trajectory in forward_trajectory:
        MMSI.append(single_trajectory[0][0])
        # 删除MMSI和日期
        single_trajectory = np.delete(single_trajectory, 0, axis=1)
        single_trajectory = np.delete(single_trajectory, 0, axis=1)
        forward_trajectory_chain.append(list(chain.from_iterable(single_trajectory)))
    PCA_forward = pca_sk.fit_transform(forward_trajectory_chain)
    k = [[MMSI[i]] + list(PCA_forward[i]) for i in range(len(MMSI))]
    print("===========forward_trajectory的PCA降维完成==========")
    return k


# 采用GMM模型对轨迹聚类，一共聚为K=7类，输入PCA_forward,返回cluster_trajectory
def clustering_trajectory(PCA_forward1):
    MMSI = []
    PCA_forward = []
    for t in PCA_forward1:
        PCA_forward.append(list(t))
    for k in PCA_forward:
        MMSI.append(k[0])
        del k[0]
    ##设置gmm函数
    gmm = GaussianMixture(n_components=4, covariance_type='full').fit(PCA_forward)
    ##训练数据
    y_pred = gmm.predict(PCA_forward)
    cluster_trajectory = []
    for i in range(len(MMSI)):
        cluster_trajectory_temp = [MMSI[i]] + PCA_forward[i] + [y_pred[i]]
        cluster_trajectory.append(cluster_trajectory_temp)
        cluster_trajectory_temp = []
    print("===============forward_trajectory的GMM聚类完成==============")
    return cluster_trajectory


#  对back_trajectory打上与forward_trajectory相同的label,传入back_trajectory与cluster_trajectory,返回back_trajectory_with_label
def back_trajectory_label(back_trajectory, cluster_trajectory):
    for i in range(len(back_trajectory)):
        for k in range(len(back_trajectory[i])):
            del back_trajectory[i][k][1]
            back_trajectory[i][k].append(cluster_trajectory[i][4])
    back_trajectory_with_label = back_trajectory
    return back_trajectory_with_label


# 对back_trajectory_with_label采用LDA降维，输入back_trajectory_with_label和back_select,返回LDA_back与LDA_select
def LDA_3(back_trajectory_with_label, back_select):
    back_trajectory_chain = []
    MMSI = []
    label = []
    for single_trajectory in back_trajectory_with_label:
        MMSI.append(single_trajectory[0][0])
        label.append(single_trajectory[0][5])
        # 删除MMSI,删除label
        for point in single_trajectory:
            del point[0]
            del point[4]
        back_trajectory_chain.append(list(chain.from_iterable(single_trajectory)))
    back1 = np.delete(back_select, 0, axis=1)
    back1 = np.delete(back1, 0, axis=1)
    back2 = []
    for k in back1:
        back2.append(list(k))
    a = list(sum(back2, []))
    lda = LinearDiscriminantAnalysis(n_components=3)
    lda.fit(back_trajectory_chain, label)
    LDA_back1 = lda.transform(back_trajectory_chain)
    LDA_select1 = lda.transform([a])
    LDA_back = [[MMSI[i]] + list(LDA_back1[i]) + [label[i]] for i in range(len(MMSI))]
    LDA_select = [[back_select[0][0]] + list(LDA_select1[0])]
    print("===========back_trajectory的LDA降维完成==========")
    return LDA_back, LDA_select


# 使用KNN分类算法，将LDA_back分类到LDP_select的类别当中，其中参数K=7，返回类别kind
def KNN_7(LDA_back1, LDA_select1):
    LDA_back = copy.deepcopy(LDA_back1)
    LDA_select = copy.deepcopy(LDA_select1)
    label = []
    for single in LDA_back:
        del single[0]
        label.append(single[3])
        del single[3]
    del LDA_select[0][0]
    k = 7
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(LDA_back, label)
    kind = clf.predict(LDA_select)
    print("=================KNN分类完成=================")
    return kind[0]


# 计算LDA_back中kind类别的轨迹与LDA_select的距离，返回他们的[MMSI,weight]
def getWeight(LDA_back1, LDA_select1, kind):
    LDA_back = copy.deepcopy(LDA_back1)
    LDA_select = copy.deepcopy(LDA_select1)
    MMSI = []
    weigth = []
    sum_weight = 0
    del LDA_select[0][0]
    LDA_select1 = numpy.array(LDA_select[0])
    for single in LDA_back:
        if single[4] == kind:
            MMSI.append(single[0])
            del single[0]
            del single[3]
            single1 = numpy.array(single)
            distance = numpy.sqrt(numpy.sum(numpy.square(LDA_select1 - single1)))
            weigth.append(1 / distance)
    for i in weigth:
        sum_weight = sum_weight + i
    weigth = weigth / sum_weight
    return list(zip(MMSI, weigth))


# 预测，输入weight和forward_trajectory,返回pre_trajectory
def get_pre_trajectory(weight, forward_trajectory):
    LAT = [0 for x in range(0, 60)]
    LON = [0 for x in range(0, 60)]
    SOG = [0 for x in range(0, 60)]
    COG = [0 for x in range(0, 60)]
    dict_weight = dict(weight)
    dict_list = list(dict_weight.keys())
    for single in forward_trajectory:
        if single[0][0] in dict_list:
            for i in range(len(single)):
                LAT[i] = LAT[i] + single[i][2] * dict_weight[single[0][0]]
                LON[i] = LON[i] + single[i][3] * dict_weight[single[0][0]]
                SOG[i] = SOG[i] + single[i][4] * dict_weight[single[0][0]]
                COG[i] = COG[i] + single[i][5] * dict_weight[single[0][0]]
    pre_trajectory = list(zip(LAT, LON, SOG, COG))
    print("===================预测轨迹成功！==================")
    return pre_trajectory

# 轨迹提取画图
def draw(back_trajectory, forward_trajectory):
    map = Basemap(llcrnrlon=-180, llcrnrlat=51, urcrnrlon=-174, urcrnrlat=54,
                  resolution='f', projection='tmerc', lat_0=53, lon_0=-178.5)
    map.drawmapboundary(fill_color='aqua')
    map.fillcontinents(color='coral', lake_color='aqua')
    map.drawcoastlines()
    map.drawmeridians(np.arange(-180, -174 + 0.001, 0.5), labels=[1, 1, 1, 1])  # 经线
    map.drawparallels(np.arange(51, 54 + 0.001, 0.5), labels=[1, 1, 1, 1])  # 纬线
    # x是经度，y是纬度
    for trajecctory in back_trajectory:
        # print(point.lon,point.lat)
        for point in trajecctory:
            x, y = map(float(point[3]), float(point[2]))
            map.plot(y, x, marker='.', color='green', markersize=1)  # 纬度放在前面，经度放在后面,绿色代表back_trajectory
    print("back_trajectory画图成功")
    for trajecctory in forward_trajectory:
        # print(point.lon,point.lat)
        for point in trajecctory:
            x, y = map(float(point[3]), float(point[2]))
            map.plot(y, x, marker='.', color='orange', markersize=1)  # 纬度放在前面，经度放在后面,橘色代表forward_trajectory
    print("forward_trajectory取轨迹画图成功")
    plt.savefig("test{0}周围轨迹提取.png".format(MMSI), dpi=1080)
    plt.show()

# 轨迹预测画图
def draw_prediction(real_trajectory,pre_trajectory,real_trajectory1):
    map = Basemap(llcrnrlon=-180, llcrnrlat=51, urcrnrlon=-174, urcrnrlat=54,
                  resolution='f', projection='tmerc', lat_0=53, lon_0=-178.5)
    map.drawmapboundary(fill_color='aqua')
    map.fillcontinents(color='coral', lake_color='aqua')
    map.drawcoastlines()
    map.drawmeridians(np.arange(-180, -174 + 0.001, 0.5), labels=[1, 1, 1, 1])  # 经线
    map.drawparallels(np.arange(51, 54 + 0.001, 0.5), labels=[1, 1, 1, 1])  # 纬线
    # x是经度，y是纬度
    for point in real_trajectory:
        x, y = map(float(point[3]), float(point[2]))
        map.plot(y, x, marker='.', color='green', markersize=1)  # 纬度放在前面，经度放在后面,绿色代表back_trajectory
    print("真实画图成功")
    for point in pre_trajectory:
        x, y = map(float(point[1]), float(point[0]))
        map.plot(y, x, marker='.', color='orange', markersize=1)  # 纬度放在前面，经度放在后面,橘色代表forward_trajectory
    print("预测取轨迹画图成功")
    plt.savefig("test{0}预测轨迹.png".format(MMSI), dpi=1080)
    plt.show()

def out():
    real_tajectory1, first_point1 = readdata()
    first_point, result_trajectory = zuobiaozhou(first_point1[0])
    S0 = getS0(first_point, result_trajectory)
    back_trajectory, forward_trajectory, back_select,forward_select = getTrajectory(S0, first_point)
    PCA_forward = PCA_3(forward_trajectory)
    cluster_trajectory = clustering_trajectory(PCA_forward)
    back_trajectory_with_label = back_trajectory_label(back_trajectory, cluster_trajectory)
    LDA_back, LDA_select = LDA_3(back_trajectory_with_label, back_select)
    kind = KNN_7(LDA_back, LDA_select)
    weight = getWeight(LDA_back, LDA_select, kind)
    pre_trajectory = get_pre_trajectory(weight, forward_trajectory)
    draw_prediction(back_select+forward_select,pre_trajectory,real_tajectory1)

out()
