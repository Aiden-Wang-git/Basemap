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



def draw(trajectory):
    map = Basemap(llcrnrlon=-180, llcrnrlat=51, urcrnrlon=-174, urcrnrlat=54,
                  resolution='f', projection='tmerc', lat_0=53, lon_0=-178.5)
    map.drawmapboundary(fill_color='aqua')
    map.fillcontinents(color='coral', lake_color='aqua')
    map.drawcoastlines()
    map.drawmeridians(np.arange(-180, -174 + 0.001, 0.5), labels=[1, 1, 1, 1])  # 经线
    map.drawparallels(np.arange(51, 54 + 0.001, 0.5), labels=[1, 1, 1, 1])  # 纬线
    # x是经度，y是纬度
    for point in trajectory:
        x, y = map(float(point[4]), float(point[3]))
        map.plot(y, x, marker='.', color='green', markersize=1)  # 纬度放在前面，经度放在后面,绿色代表back_trajectory
    print("back_trajectory画图成功")
    plt.show()


def get_Trajectory():
    db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # SQL 查询语句
    sql_S0 = "SELECT * FROM aispoints WHERE  lat BETWEEN 51 AND 54 AND lon BETWEEN -180 AND -174 ;";
    try:
        # 执行SQL语句,获得周围目标船舶的航迹
        cursor.execute(sql_S0)
        tajectory = cursor.fetchall()
    except:
        print
        "Error: unable to fecth data"
    return tajectory


tajectory = get_Trajectory()
draw(tajectory)
