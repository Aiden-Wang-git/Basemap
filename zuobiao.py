# -*- coding: utf-8 -*-
# @Time : 2019/6/6 8:13
# @Author : WangPeng
from sympy import *
from math import radians, cos, sin, asin, sqrt


class Get_new_gps():
    def __init__(self):
        # 地球半径
        self.R = 6371 * 1000
        pass

    """计算两点间距离"""

    def geodistance(self, lng1, lat1, lng2, lat2):
        """

        :param lng1: 120.12802999999997
        :param lat1: 30.28708,115
        :param lng2: 115.86572000000001
        :param lat2: 28.7427
        :return: 两点间距离
        """
        lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
        dlon = lng2 - lng1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        distance = 2 * asin(sqrt(a)) * self.R  # 地球平均半径，6371km
        distance = round(distance, 3)
        return distance

    """计算点经纬度北500米的点的经纬度"""

    def get_new_lat(self, lng1, lat1, dist=500):
        """

        :param lng1: 120.12803
        :param lat1: 20.28708
        :param dist:
        :return: (120.12803, 20.291576608029594)
        """
        lat2 = 180 * dist / (self.R * pi) + lat1
        return (lng1, lat2)

    """计算点经纬度正东500米点的经纬度"""

    def get_new_lng(self, lng1, lat1, dist=500):
        """

        :param lng1: 116.55272514141352
        :param lat1: 30.28708
        :param dist:
        :return: (116.55272514141352,30.28708)
        """
        lng2 = 180 * dist / (self.R * pi * cos(radians(lat1))) + lng1
        return (lng2, lat1)

    """计算点东北方向与正东某夹角某距离的经纬度"""

    def get_new_lng_angle(self, lng1, lat1, dist=500, angle=30):
        """

        :param lng1:116.55272514141352
        :param lat1:30.28708
        :param dist:指定距离
        :param angle:指定角度
        :return:（0.0091871843081617/pi + 116.498079 0.0122339171779312/pi + 39.752304）
        """
        lat2 = 180 * dist*sin(radians(angle)) / (self.R * pi) + lat1
        lng2 = 180 * dist*cos(radians(angle)) / (self.R * pi * cos(radians(lat1))) + lng1
        return (lng2, lat2)
functions = Get_new_gps()

if __name__ == '__main__':
    functions = Get_new_gps()
    lng1, lat1 = [116.498079, 39.752304]
    # 计算正北的点
    lng2, lat2 = functions.get_new_lat(lng1, lat1)
    # 计算正东的点
    lng3, lat3 = functions.get_new_lng(lng1, lat1)
    # 计算该点与正东逆时针夹角45度，距离500米点的经纬度值
    lng4, lat4 = functions.get_new_lng_angle(lng1, lat1, dist=500, angle=45)
    print("原始点的经纬度坐标", lng1, lat1)
    print("正北500米坐标点为%f,%f,距离计算为%f米" % (lng2, lat2, functions.geodistance(lng1, lat1, lng2, lat2)))
    print("正东500米坐标点为%f,%f,距离计算为%f米" % (lng3, lat3, functions.geodistance(lng1, lat1, lng3, lat3)))
    print("东北方夹角，距离500米坐标点为%f,%f,距离计算为%f米" % (float(lng3), float(lat3), functions.geodistance(lng1, lat1, lng3, lat3)))
    print(functions.geodistance(lng3, lat3, lng4, lat4))
