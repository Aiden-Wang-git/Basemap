from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from TrajectoryCluster.aisPoint import AIS
from sqlalchemy import and_
from sqlalchemy import func
from TrajectoryCluster.trajectory import Trajectory
import time
import matplotlib.pyplot as plt
import pandas as pd
from TrajectoryCluster.dtw import DTW, DTWSpatialDis, DTWCompare, DTW1, DTWSpatialDisCOM
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # 计算 轮廓系数，CH 指标，DBI
from TrajectoryCluster.myHausdorff import hausdorff
from pylab import mpl
from geopy.distance import geodesic
from math import radians, cos, sin, asin, sqrt

# ========================全文种使用到的参数==============================

# ===================从数据库中查询数据的条件=============================
top = 33.65
bottom = 33.55
left = -118.30
right = -118.20
begin = '2017-01-01'
end = '2018-01-01'

# ===================航迹预处理的条件=============================
# 一条航迹中至少包含AIS点数目
min_num_in_trajectory = 35
# 相邻AIS点之间的时间阈值,单位S
max_time_between_AIS = 1800
# ==================最大航速35节，通过数据估计的====================
max_sog = 33

# ====================画图时，如遇中文显示问题可加入以下代码============
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# ==========================================原始航迹提取===============================
# 返回的结果是个字典，key为MMSI，value为按时间排序的AIS点
def getRawTrajectory(top, bottom, left, right, begin, end):
    # 连接数据库
    engine = create_engine("mysql+pymysql://root:123456@localhost:3306/ais?charset=utf8")
    DbSession = sessionmaker(bind=engine)
    session = DbSession()
    datas = session.query(AIS).filter(
        # 取值范围(33.55N,33.65N),(-118.30W,-118.20W),'2017-01-01'~'2018-01-01'
        and_(AIS.LAT >= bottom,
             AIS.LAT <= top,
             AIS.LON >= left,
             AIS.LON <= right,
             AIS.SOG >= 1,
             AIS.BaseDateTime >= begin,
             AIS.BaseDateTime <= end)).order_by(AIS.BaseDateTime).all()
    session.close()
    trajectories = {}
    for data in datas:
        if trajectories.__contains__(data.MMSI):
            trajectories[data.MMSI].append(data)
        else:
            trajectories[data.MMSI] = [data]
    print(f"从数据库中提取轨迹，共有不同类别MMSI:{len(trajectories)}个，AIS点：{len(datas)}个")
    return trajectories


# ==========================================对原始航迹进行分段及删除短航迹================
# 删除AIS点数目小于35个的短航迹，对相邻间隔大于30min的航迹分段
def process1(trajectories):
    trajectories_process1 = {}
    count = 0
    for key in trajectories:
        trajectory = trajectories[key]
        split_time = [0]
        # 原航迹太短，直接删除
        if len(trajectory) < min_num_in_trajectory:
            continue
        else:
            # 找到分割点
            for i in range(len(trajectory) - 1):
                if (trajectory[i + 1].BaseDateTime - trajectory[i].BaseDateTime).seconds > max_time_between_AIS:
                    split_time.append(i + 1)
            # 往新的字典中添加数据
            if len(split_time) > 1:
                for i in range(len(split_time) - 1):
                    if split_time[i + 1] - split_time[i] >= min_num_in_trajectory:
                        trajectories_process1[key + '-' + str(i)] = trajectory[split_time[i]:split_time[i + 1]]
                        count += len(trajectories_process1[key + '-' + str(i)])
            else:
                trajectories_process1[key] = trajectory
                count += len(trajectories_process1[key])
    print(f"经过process1之后，共有航迹：{len(trajectories_process1)}条,AIS点：{count}个")
    return trajectories_process1


# 处理位置异常点，以及航速大于35节的点
def process2(trajectories_process1):
    count1 = 0
    count2 = 0
    for key in trajectories:
        trajectory = trajectories[key]
        for i in range(1, len(trajectory) - 1):
            if trajectory[i].SOG > max_sog:
                count1 += 1
            elif getDistance(trajectory[i], trajectory[i + 1]) / (
                    trajectory[i + 1].BaseDateTime - trajectory[i].BaseDateTime).seconds * 3600 > 1.852 * max_sog \
                    or getDistance(trajectory[i - 1], trajectory[i]) / (
                    trajectory[i].BaseDateTime - trajectory[i - 1].BaseDateTime).seconds * 3600 > 1.852 * max_sog:
                count2 += 1
    print(f"经过process2之后,AIS点超速count1:{count1}个")
    print(f"经过process2之后,AIS点异常count2:{count2}个")
    return trajectories_process1


# =========================================工具函数====================================
# 1.根据两个AIS点，计算两者之间的实际距离
def getDistance(pointA, pointB):
    # 这个方法太慢
    # return geodesic((pointA.LAT, pointA.LON), (pointB.LAT, pointB.LON)).km
    lng1 = pointA.LON
    lat1 = pointA.LAT
    lng2 = pointB.LON
    lat2 = pointB.LAT
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 3)
    return distance


# ==================================主函数部分==========================================
trajectories = getRawTrajectory(top=top, bottom=bottom, left=left, right=right, begin=begin, end=end)
trajectories_process1 = process1(trajectories=trajectories)
trajectories_process2 = process2(trajectories_process1=trajectories_process1)
