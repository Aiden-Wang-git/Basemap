import MySQLdb
from math import radians, cos, sin, asin, sqrt
import numpy as np
from geopy.distance import geodesic
import time,datetime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import interpolate
import pandas as pd

# 预测船舶ID,预测点时间
MMSI = "366940480"
Basedatetime = "2017-01-04 15:36:07"


# 读取预测船舶的信息，返回真实航迹real_trajectory以及第一个点的信息first_point1
def readdata():
    db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # SQL 查询语句
    sql_truth = "SELECT * FROM aispoints WHERE  MMSI = '{0}' order by BaseDateTime ;".format(MMSI)
    sql_truth_point = "SELECT * FROM aispoints WHERE  MMSI = '{0}'  and BaseDateTime = '{1}';".format(MMSI, Basedatetime)
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
    # SQL 查询语句,查询起始点附近0.5°的船只信息
    sql_result = "SELECT * FROM aispoints WHERE lat BETWEEN {0} AND {1} AND lon BETWEEN {2} AND {3} ;".format(first_LAT-0.5,first_LAT+0.5,first_LON-0.5,first_LON+0.5)
    try:
        # 执行SQL语句,获得预测真实航迹
        cursor.execute(sql_result)
        result_trajectory2= cursor.fetchall()
    except:
        print
        "Error: unable to fecth data"
    # # 将起始点坐和其余数据点标转换
    # R =[[cos(first_COG),-sin(first_COG)],[sin(first_COG),cos(first_COG)]]
    # first_point1[4],first_point1[3] = np.dot(R,np.transpose([first_LON,first_LAT]))
    # for i in range(len(result_trajectory1)):
    #     result_trajectory1[i]=list(result_trajectory1[i])
    #     result_trajectory1[i][4],result_trajectory1[i][3]=np.dot(R,np.transpose([result_trajectory1[i][4],result_trajectory1[i][3]]))
    #比较距离以及航向的
    result_trajectory1 = list(result_trajectory2)
    result_trajectory = [[]]
    first_point = [first_point1[1],first_point1[2],first_point1[3],first_point1[4],first_point1[5],first_point1[6]]
    for i in range(len(result_trajectory1)):
        if abs(first_point[5] - result_trajectory1[i][6])<45 or abs(first_point[5] - result_trajectory1[i][6])>315:
            if abs(first_point[4]-result_trajectory1[i][5])<3:
                result_trajectory.append([result_trajectory1[i][1],result_trajectory1[i][2],result_trajectory1[i][3],result_trajectory1[i][4],result_trajectory1[i][5],result_trajectory1[i][6]])
    del (result_trajectory[0])
    return first_point,result_trajectory

# 根据经纬度确定两点之间的距离
def getDistanceDiffrentLat(Lat1,Lat2,Lon):
    distance = geodesic((Lat1, Lon), (Lat2, Lon)).nm
    return distance
def getDistanceDiffrentLon(Lon1,Lon2,Lat):
    distance = geodesic((Lat, Lon1), (Lat, Lon2)).nm
    return distance
def getDistance(pointA=[],pointB = []):
    distance = geodesic((pointA[2], pointA[3]), (pointB[2], pointB[3])).nm
    return distance

# 输入预测点和周围数据点，返回初始类S0
def getS0(first_point, result_trajectory):
    S0 = [[]]
    del (S0[0])
    for i in range(len(result_trajectory)):
        if getDistance(first_point,result_trajectory[i])<2:
            S0.append(result_trajectory[i])
    temp = {}
    for MMSI, BaseDateTime, LAT, LON, SOG, COG in S0:
        if MMSI not in temp:  # we see this key for the first time
            temp[MMSI] = (MMSI, BaseDateTime, LAT, LON, SOG, COG)
        else:
            # 找出相同MMSI中距离预测点最近的那个点
            if getDistance(first_point,[MMSI, BaseDateTime, LAT, LON, SOG, COG]) < getDistance(first_point, temp[MMSI]):
                temp[MMSI] = (MMSI, BaseDateTime, LAT, LON, SOG, COG)
    # S0 = temp.values()
    return S0

# 轨迹提取，根据S0，分别向后和向前提取60个点，时间间隔为30S，向前向后都是30min,返回back_trajectory和forward_trajectory
def getTrajectory(S0):
    back_trajectory =[]
    forward_trajectory =[]
    print("SO中一共会有",len(S0),"条航迹")
    for i in range(len(S0)):
        db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8')
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # SQL 查询语句
        sql_S0 = "SELECT * FROM aispoints WHERE  MMSI = '{0}' and  lat BETWEEN {1} AND {2} AND lon BETWEEN {3} AND {4} order by BaseDateTime ;".format(S0[i][0],S0[i][2]-2,S0[i][2]+2,S0[i][3]-2,S0[i][3]+2)
        try:
            # 执行SQL语句,获得周围目标船舶的航迹
            cursor.execute(sql_S0)
            S0_tajectory = cursor.fetchall()
        except:
            print
            "Error: unable to fecth data"
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
        X = pd.date_range(start=first_time-datetime.timedelta(minutes=30),periods=121,freq='30S')
        X = list(X)
        XStamp = []
        for h in range(len(X)):
            XStamp.append(int(X[h].timestamp()))
        # X = np.linspace(time-datetime.timedelta(seconds=30),time+datetime.timedelta(seconds=30),121)
        f = interpolate.interp1d(dateStamp, [LAT,LON,SOG,COG], kind="nearest")
        # f_LON = interpolate.interp1d(date, LON, kind="nearest")
        # f_SOG = interpolate.interp1d(date, SOG, kind="nearest")
        # f_COG = interpolate.interp1d(date, COG, kind="nearest")
        # LATS = f_LAT(X)
        # LONS = f_LON(X)
        # SOGS = f_SOG(X)
        # COGS = f_COG(X)
        LATS,LONS,SOGS,COGS = f(XStamp)
        back = []
        forward = []
        for j in range(0,60):
            back.append([S0[i][0],X[j],LATS[j],LONS[j],SOGS[j],COGS[j]])
        back_trajectory.append(back)
        for m in range(61,121):
            forward.append([S0[i][0],X[m],LATS[m],LONS[m],SOGS[m],COGS[m]])
        forward_trajectory.append(forward)
        print("轨迹",i,"完成！")
    # print("向后提取轨迹：")
    # print(back_trajectory)
    # print("向前提取轨迹：")
    # print(forward_trajectory)
    return back_trajectory,forward_trajectory

# 画图
def draw(back_trajectory,forward_trajectory):
    map = Basemap(llcrnrlon = -180, llcrnrlat = 51, urcrnrlon = -174, urcrnrlat = 54,
                resolution = 'f', projection = 'tmerc',lat_0=53,lon_0=-178.5)
    map.drawmapboundary(fill_color='aqua')
    map.fillcontinents(color='coral', lake_color='aqua')
    map.drawcoastlines()
    map.drawmeridians(np.arange(-180, -174 + 0.001, 0.5), labels=[1, 1, 1, 1])    # 经线
    map.drawparallels(np.arange(51, 54 + 0.001, 0.5), labels=[1, 1, 1, 1])    # 纬线
    # x是经度，y是纬度
    for trajecctory in back_trajectory:
        # print(point.lon,point.lat)
        for point in trajecctory:
            x, y = map(float(point[3]),float(point[2]))
            map.plot(y, x, marker='.', color='green', markersize=1) # 纬度放在前面，经度放在后面,绿色代表back_trajectory
    print("back_trajectory画图成功")
    for trajecctory in forward_trajectory:
        # print(point.lon,point.lat)
        for point in trajecctory:
            x, y = map(float(point[3]),float(point[2]))
            map.plot(y, x, marker='.', color='orange', markersize=1) # 纬度放在前面，经度放在后面,橘色代表forward_trajectory
    print("forward_trajectory取轨迹画图成功")
    plt.savefig("test{0}周围轨迹提取.png".format(MMSI), dpi = 1080)
    plt.show()

def out():
    real_tajectory1, first_point1 = readdata()
    first_point, result_trajectory = zuobiaozhou(first_point1[0])
    S0 = getS0(first_point,result_trajectory)
    back_trajectory,forward_trajectory=getTrajectory(S0)
    draw(back_trajectory,forward_trajectory)
out()