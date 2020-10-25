import copy
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



def draw(trajectory):
    map = Basemap(llcrnrlon=-118.3, llcrnrlat=33.7, urcrnrlon=-118.2, urcrnrlat=33.8,resolution='f')
    map.drawmapboundary(fill_color='aqua')
    map.fillcontinents(color='coral', lake_color='aqua')
    map.drawcoastlines()
    # map.drawmeridians(np.arange(-140, -110 + 0.001, 1), labels=[1, 1, 1, 1])  # 经线
    # map.drawparallels(np.arange(30, 50 + 0.001, 1), labels=[1, 1, 1, 1])  # 纬线
    # x是经度，y是纬度
    i = 0
    for point in trajectory:
        x, y = map(float(point[4]), float(point[3]))
        map.plot(x, y, marker='o', color='m')  # 纬度放在前面，经度放在后面,绿色代表back_trajectory
        print("画好第",i,"条")
        i+=1
    print("trajectory画图成功")
    plt.savefig("test_AIS.png", dpi=1080)
    plt.show()


def get_Trajectory():
    db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # SQL 查询语句
    sql_S0 = "SELECT * FROM ais_2018_01_02 WHERE MMSI = '366760650' AND BaseDateTime > '2018-01-02 12:39:49' AND BaseDateTime<'2018-01-02 12:52:29';"
    try:
        # 执行SQL语句,获得周围目标船舶的航迹
        cursor.execute(sql_S0)
        tajectory = cursor.fetchall()
    except:
        print
        "Error: unable to fecth data"
    return tajectory

def out():
    tajectory = get_Trajectory()
    print("数据收集完成")
    print(len(tajectory))
    draw(tajectory)

out()