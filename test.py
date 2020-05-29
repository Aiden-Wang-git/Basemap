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
    def get_new_lng_angle(self, lat1,lng1,  dist=1000, angle=30): #给定经度、纬度、距离、航向，得到纬度、经度
        """

        :param lng1:116.55272514141352
        :param lat1:30.28708
        :param dist:指定距离
        :param angle:指定角度
        :return:（0.0091871843081617/pi + 116.498079 0.0122339171779312/pi + 39.752304）
        """
        lat2 = 180 * dist * sin(radians(angle+90)) / (self.R * pi) + lat1
        lng2 = -180 * dist * cos(radians(angle+90)) / (self.R * pi * cos(radians(lat1))) + lng1
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
    map = Basemap(llcrnrlon = -180, llcrnrlat = 51, urcrnrlon = -174, urcrnrlat = 55,
                resolution = 'f', projection = 'tmerc', lat_0 = 61, lon_0 = 2)
    map.drawmapboundary(fill_color='aqua')
    map.fillcontinents(color='coral', lake_color='aqua')
    map.drawcoastlines()
    # x是经度，y是纬度
    for point in list:
        # print(point.lon,point.lat)
        x, y = map(float(point[3]),float(point[2]))
        print(point)
        map.plot(y, x, marker='.', color='green', markersize=1)#纬度放在前面，经度放在后面,绿色代表预测点
    print("预测点画图成功")
    index = 0
    for point in truth:
        index+=1
        if index>150:
            break
        print(point)
        # print(point.lon,point.lat)
        x, y = map(float(point[4]),float(point[3]))
        map.plot(y, x, marker='.', color='m', markersize=1)#真实点用紫色标志
    print("真实点画图成功")
    plt.savefig("test{0}.png".format(filename))
    plt.show()


#寻找半径内所有点
def getDistance(point_now=[],list = [[]]):
    li=[]
    for point in list:
        if geodesic((point_now[2],point_now[3]), (point[3],point[4])).m < 2000:#计算两个坐标直线距离
            li.append(point)
    return li

#预测，返回预测的航向、航速和预测点的时间、纬度、经度
def getPredictionFirst(point_now=[],list = [[]]):
    count = 0
    sumheading = 0
    sumspeed = 0
    for num in range(len(list)):
        if (abs(list[num][7]-point_now[5])<45)or(abs(list[num][7]-point_now[5])>315): #寻找满足航向要求的点
            count+=1
            sumheading+=list[num][7]
            sumspeed+=list[num][5]
    print("count={0}".format(count))
    if(count!=0):
        avghead = float(sumheading/count)
        avgspeed = float(sumspeed/count)
    else:
        avghead = point_now[5]
        avgspeed = point_now[4]
    # timeArray = time.strptime(point_now[2], "%Y-%m-%d %H:%M:%S")
    timeStamp = time.mktime(point_now[1].timetuple())
    timeStampnew = timeStamp + int(1000/(avgspeed*0.51444))
    timeArray = time.localtime(timeStampnew)
    nexttime_old = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    nexttime = datetime.datetime.strptime(nexttime_old,"%Y-%m-%d %H:%M:%S")
    lat_new,lon_new = functions.get_new_lng_angle(point_now[2],point_now[3],1000,avghead)
    return nexttime,round(lat_new,6),round(lon_new,6),round(avgspeed,6),round(avghead,6)

functions = Get_New_Gps()
# 船只ID
filename = "338626000"

def output():
    db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # SQL 查询语句
    sql_truth = "SELECT * FROM aispoints WHERE lat BETWEEN 50 AND 55 AND lon BETWEEN -180 AND -150 AND MMSI = '{0}' order by BaseDateTime ;".format(filename)
    try:
        # 执行SQL语句
        cursor.execute(sql_truth)
        # 获取某一条船只记录列表
        results_truth = cursor.fetchall()
    except:
        print
        "Error: unable to fecth data"
    prediction = [[]]
    prediction[0] = [results_truth[50][1],results_truth[50][2],results_truth[50][3],results_truth[50][4],results_truth[50][5],results_truth[50][7]]
    print(prediction[0])
    for i in range(0,100):
        sql_area = "SELECT * FROM aispoints WHERE lat BETWEEN {0} AND {1} AND lon BETWEEN {2} AND {3} AND MMSI<>{4};".format(float(prediction[i][2])-0.5,float(prediction[i][2])+0.5,float(prediction[i][3])-0.5,float(prediction[i][3])+0.5,filename)
        try:
            # 执行SQL语句
            cursor.execute(sql_area)
            # 获取区域内所有记录列表
            results_area = cursor.fetchall()
            # 执行SQL语句
        except:
            print
            "Error: unable to fecth data"
        list1 = getDistance(prediction[i],results_area)
        prediction_time,prediction_lat,prediction_lon,prediction_speed,prediction_head = getPredictionFirst(prediction[i],list1)
        prediction.append([prediction[0][0],prediction_time,prediction_lat,prediction_lon,prediction_speed,prediction_head])
        print("预测第{0}个点".format(i+1))
    # 关闭数据库连接
    db.close()
    draw(prediction,results_truth)
    # for x in result:
    #         print(x.lon,x.lat,x.speed,x.heading,x.time,x.ID)
output()
