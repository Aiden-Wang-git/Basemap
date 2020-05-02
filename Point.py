class Point:
    def __init__(self,ID,time,heading,speed,lat,lon): #编号、时间、航向、航速、纬度、经度
        self.ID = ID
        self.time = time
        self.heading = heading
        self.speed = speed
        self.lat = lat
        self.lon = lon

point = Point('ID', '时间', '航向','速度', 'LA', 'LON')
# # # print(point)