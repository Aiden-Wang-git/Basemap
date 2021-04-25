import math


class Trajectory:
    points = []
    count = 0
    MMSI = "null"
    D = 0.0015  # 航迹压缩算法的容差
    index = []  # 用于表示AIS点在轨迹上的索引
    error = 0  # 用于统计删除点的误差合计值
    deleteNum = 0  # 用于表示共删除了多少个点
    label = -1  # 用于表示航迹属于哪个类
    r = 0.01  # 用于对航迹中环状片段进行删除
    deleteCircleNum = 0  # 用于统计在去环时删除的点个数

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
        if right == left + 1:
            return
        distance = self.getDistance(left, p1, p2, right)
        dmax = max(distance)
        # 删除阈值范围内的点
        if dmax <= self.D:
            for i in range(right - left - 1):
                del self.points[left + 1]
                self.count = self.count - 1
            self.error += sum(distance)
            self.deleteNum += len(distance)
        else:
            # middle表示距离最远的点
            middle = self.points[distance.index(dmax) + left + 1]
            self.compress(p1, middle)
            self.compress(middle, p2)

    # 计算D-P算法的误差
    def getDistance(self, left, p1, p2, right):
        A = p1.LAT - p2.LAT
        B = p2.LON - p1.LON
        C = p1.LON * p2.LAT - p2.LON * p1.LAT
        distance = []
        # 计算中点到直线的距离
        for i in range(left + 1, right):
            d = abs(A * self.points[i].LON + B * self.points[i].LAT + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2))
            distance.append(d)
        return distance

    # 计算去环状片段的误差
    def getDistanceCircle(self, deleteSectionAll):
        distance = 0
        for deleteSection in deleteSectionAll:
            left = deleteSection[0]
            right = deleteSection[1]
            p1 = self.points[left]
            p2 = self.points[right]
            distance += sum(self.getDistance(left, p1, p2, right))
        return distance

    # 轨迹去环状片段
    def deleteCircle(self):
        # 用于保存去环后的轨迹点下标
        index = [0]
        for i in range(1, self.count - 1):
            if ((self.points[i].LAT - self.points[index[len(index) - 1]].LAT) ** 2 + (
                    self.points[i].LON - self.points[index[len(index) - 1]].LON) ** 2) ** 0.5 > self.r:
                index.append(i)
                continue
            self.deleteCircleNum = self.deleteCircleNum + 1
            print("发生相邻航迹点距离小于r的情况，删除该点")
        index.append(self.count - 1)
        self.deleteNum +=self.deleteCircleNum
        # 用于保存删除点下标的左右边界
        deleteSection = []
        for i in range(len(index) - 1):
            if index[i + 1] - index[i] == 1:
                continue
            deleteSection.append([index[i], index[i + 1]])
        self.error += self.getDistanceCircle(deleteSection)
        newPoints = []
        for i in index:
                newPoints.append(self.points[i])
        self.points = newPoints
        self.count = len(self.points)

    def __init__(self, MMSI):
        self.MMSI = MMSI
        self.points = []
