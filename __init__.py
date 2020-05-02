import math
from sympy import *
from geopy.distance import geodesic
from math import radians, cos, sin, asin, sqrt

class Get_New_Gps():
    def __init__(self):
        # 地球半径
        self.R = 6371 * 1000
        pass
    def get_new_lng_angle(self, lng1, lat1, dist=500, angle=30):
        """

        :param lng1:116.55272514141352
        :param lat1:30.28708
        :param dist:指定距离
        :param angle:指定角度
        :return:（0.0091871843081617/pi + 116.498079 0.0122339171779312/pi + 39.752304）
        """
        lat2 = 180 * dist * sin(radians(angle)) / (self.R * pi) + lat1
        lng2 = 180 * dist * cos(radians(angle)) / (self.R * pi * cos(radians(lat1))) + lng1
        return (lat2,lng2 )
functions = Get_New_Gps()
lat = 30.28708
lon = 120.12802999999997
lng1, lat1 = [120.12802999999997, 30.28708]
lat_new,lon_new = functions.get_new_lng_angle(lng1, lat1, dist=500, angle=75)
print(geodesic((lat1,lng1),(lat_new,lon_new)).m)
# lat_new = 30.28708+ 360*400*math.sin(((45+90)/180)*math.pi)/(6371000*math.pi)
# lon_new = 120.12802999999997+400*math.cos(((45+90)/180)*math.pi)/(2*math.pi*math.cos(30.28708)*6371000)*360
# print(geodesic((39.752304,116.498079), (39.756801,116.498079)).m)
# 返回 447.2497993542003 千米