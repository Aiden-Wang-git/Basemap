import MySQLdb
import matplotlib.pyplot as plt
import json
import os, sys
import math
from geopy.distance import geodesic
# 打开数据库连接
db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8' )

# 使用cursor()方法获取操作游标
cursor = db.cursor()
path = r"C:\Users\28586\Desktop\新建文件夹"
Path1 = os.listdir(path)
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
                    date=path1.split('_')[0][0:4]+'-'+path1.split('_')[0][4:6]+'-'+path1.split('_')[0][6:8]+' '+path1.split('_')[0][8:10]+':'+path1.split('_')[0][10:12]+':00'
                    sql = "INSERT INTO points(MMSI,time,lat,lon,speed,course) VALUES('%s','%s',%s,%s,%s,%s)"%(str(x["SHIP_ID"]),date,float(x["LAT"]),float(x["LON"]),float(x["SPEED"]),float(x["COURSE"]))
                    try:
                       # 执行sql语句
                       cursor.execute(sql)
                       # 提交到数据库执行
                       db.commit()
                    except:
                       # Rollback in case there is any error
                       print("插入错误！")
                       db.rollback()
        print(path1)
# 关闭数据库连接
db.close()

