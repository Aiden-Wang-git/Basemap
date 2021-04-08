import math
class Trajectory:
    points = []
    count = 0
    MMSI = "null"
    D = 0.0015  # 航迹压缩算法的容差
    index = []  # 用于表示AIS点在轨迹上的索引

    def add_point(self, point):
        self.points.append(point)
        self.count = self.count + 1

    def setMMSI(self, MMSI):
        self.MMSI = MMSI

    def getLength(self):
        return len(self.points)

    # 轨迹压缩
    def compress(self, p1, p2):
        self.index = list(range(self.count))
        left = self.points.index(p1)
        right = self.points.index(p2)
        if right == left+1:
            return
        A = p1.LAT - p2.LAT
        B = p2.LON - p1.LON
        C = p1.LON*p2.LAT - p2.LON*p1.LAT
        distance = []
        # 计算中点到直线的距离
        for i in range(left+1, right):
            d = abs(A*self.points[i].LON + B*self.points[i].LAT + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2))
            distance.append(d)
        dmax = max(distance)
        if dmax <= self.D:
            for i in range(right-left-1):
                del self.points[left+1]
        else:
            # middle表示距离最远的点
            middle = self.points[distance.index(dmax) + left + 1]
            self.compress(p1, middle)
            self.compress(middle, p2)

    def __init__(self, MMSI):
        self.MMSI = MMSI
        self.points = []
