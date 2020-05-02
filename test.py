import time,datetime
import MySQLdb
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import json
import os, sys
import math
from geopy.distance import geodesic
from sympy import *
from geopy.distance import geodesic
from math import radians, cos, sin, asin, sqrt

class Get_New_Gps():
    def __init__(self):
        # 地球半径
        self.R = 6371 * 1000
        pass
    def get_new_lng_angle(self, lng1, lat1, dist=500, angle=30): #给定经度、纬度、距离、航向，得到纬度、经度
        """

        :param lng1:116.55272514141352
        :param lat1:30.28708
        :param dist:指定距离
        :param angle:指定角度
        :return:（0.0091871843081617/pi + 116.498079 0.0122339171779312/pi + 39.752304）
        """
        lat2 = 180 * dist * sin(radians(angle+90)) / (self.R * pi) + lat1
        lng2 = 180 * dist * cos(radians(angle+90)) / (self.R * pi * cos(radians(lat1))) + lng1
        return (lat2,lng2 )


class Point:
    def __init__(self,ID,time,heading,speed,lat,lon): #编号、时间、航向、航速、纬度、经度
        self.ID = ID
        self.time = time
        self.heading = heading
        self.speed = speed
        self.lat = lat
        self.lon = lon

#读取数据点到内存
def resolveJson(path):
    Path1 = os.listdir(path)
    li = []
    truth = []
    for path1 in Path1:
        Path2 = os.listdir(os.path.join(path,path1))
        for path2 in Path2:
            txt = os.path.join(path,path1,path2)
            if os.path.getsize(txt)==0:
                continue
            file = open(txt, "rb")
            fileJsons = json.load(file)
            data = fileJsons["data"]
            fileJson = data["rows"]
            if data["rows"]:
                for x in fileJson:
                    # print(path1,path2)
                    point = Point(x["SHIP_ID"], path1.split('_')[0], x["HEADING"], x["SPEED"], x["LAT"], x["LON"])
                    li.append(point)
                    if point.ID == "155418667668356":
                        truth.append(point)
        print(path1)
    print("共有点的数目:",len(li))
    return li,truth

#画图
def draw(list = [],truth = []):
    map = Basemap(llcrnrlon = -10, llcrnrlat = 58, urcrnrlon = 10, urcrnrlat = 65,
                resolution = 'i', projection = 'tmerc', lat_0 = 61, lon_0 = 2)
    map.drawmapboundary(fill_color='aqua')
    map.fillcontinents(color='coral', lake_color='aqua')
    map.drawcoastlines()
    # x是经度，y是纬度
    for point in list:
        # print(point.lon,point.lat)
        x, y = map(float(point[4]),float(point[3]))
        map.plot(x, y, marker='.', color='green')
    print("预测点画图成功")
    for point in truth:
        # print(point.lon,point.lat)
        x, y = map(float(point[4]),float(point[3]))
        map.plot(x, y, marker='.', color='lime')
    print("真实点画图成功")
    plt.show()
    plt.savefig('test.png')


#寻找半径内所有点
def getDistance(point_now=[],list = [[]]):
    li=[]
    for point in list:
        if geodesic((point_now[3],point_now[4]), (point[3],point[4])).m < 500:#计算两个坐标直线距离
            li.append(point)
    return li

#预测，返回预测的航向、航速和预测点的时间、纬度、经度
def getPredictionFirst(point_now=[],list = [[]]):
    count = 0
    sumheading = 0
    sumspeed = 0
    for num in range(len(list)):
        if (abs(list[num][6]-point_now[6])<45)or(abs(list[num][6]-point_now[6])>270): #寻找满足航向要求的点
            count+=1
            sumheading+=list[num][6]
            sumspeed+=list[num][5]
    print("预测一个点")
    avghead = float(sumheading/count)
    avgspeed = float(sumspeed/count)
    # timeArray = time.strptime(point_now[2], "%Y-%m-%d %H:%M:%S")
    timeStamp = time.mktime(point_now[2].timetuple())
    timeStampnew = timeStamp + int(1000*500/avgspeed)
    timeArray = time.localtime(timeStampnew)
    nexttime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    lat_new,lon_new = functions.get_new_lng_angle(point_now[4],point_now[3],500,avghead)
    return avghead,avgspeed,nexttime,lat_new,lon_new

functions = Get_New_Gps()

def output():
    db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8')

    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    # SQL 查询语句
    sql_area = "SELECT * FROM points WHERE lat BETWEEN 58 AND 65 AND lon BETWEEN -10 AND 10;"
    sql_truth = "SELECT * FROM points WHERE lat BETWEEN 58 AND 65 AND lon BETWEEN -10 AND 10 AND MMSI = '252576';"
    try:
        # 执行SQL语句
        cursor.execute(sql_area)
        # 获取区域内所有记录列表
        results_area = cursor.fetchall()
        # 执行SQL语句
        cursor.execute(sql_truth)
        # 获取某一条船只记录列表
        results_truth = cursor.fetchall()
    except:
        print
        "Error: unable to fecth data"
    # 关闭数据库连接
    db.close()
    prediction = [[]]
    prediction[0] = results_truth[0]
    print(prediction[0])
    for i in range(0,51):
        list1 = getDistance(prediction[i],results_area)
        prediction_head,prediction_speed,prediction_time,prediction_lat,prediction_lon = getPredictionFirst(prediction[i],list1)
        prediction.append([prediction[0][1],prediction_time,prediction_head,prediction_speed,prediction_lat,prediction_lon])
        print([prediction[0][1],prediction_time,prediction_head,prediction_speed,prediction_lat,prediction_lon])
    draw(results_truth,prediction)
    # for x in result:
    #         print(x.lon,x.lat,x.speed,x.heading,x.time,x.ID)
output()
