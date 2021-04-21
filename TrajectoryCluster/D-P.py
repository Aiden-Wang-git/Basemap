# -*- coding:utf-8 -*-
"""
道格拉斯算法的实现
程序需要安装shapely模块
"""
import math
from shapely import wkt, geometry
import matplotlib.pyplot as plt


class Point:
    """点类"""
    x = 0.0
    y = 0.0
    index = 0  # 点在线上的索引

    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index


class Douglas:
    """道格拉斯算法类"""
    points = []
    D = 1  # 容差

    def readPoint(self):
        """生成点要素"""
        g = wkt.loads("LINESTRING(1 4,2 3,4 2,6 6,7 7,8 6,9 5,10 10)")
        coords = g.coords
        for i in range(len(coords)):
            self.points.append(Point(coords[i][0], coords[i][1], i))

    def compress(self, p1, p2):
        """具体的抽稀算法"""
        swichvalue = False
        # 一般式直线方程系数 A*x+B*y+C=0,利用点斜式,分母可以省略约区
        # A=(p1.y-p2.y)/math.sqrt(math.pow(p1.y-p2.y,2)+math.pow(p1.x-p2.x,2))
        A = (p1.y - p2.y)
        # B=(p2.x-p1.x)/math.sqrt(math.pow(p1.y-p2.y,2)+math.pow(p1.x-p2.x,2))
        B = (p2.x - p1.x)
        # C=(p1.x*p2.y-p2.x*p1.y)/math.sqrt(math.pow(p1.y-p2.y,2)+math.pow(p1.x-p2.x,2))
        C = (p1.x * p2.y - p2.x * p1.y)

        m = self.points.index(p1)
        n = self.points.index(p2)
        distance = []
        middle = None

        if (n == m + 1):
            return
        # 计算中间点到直线的距离
        for i in range(m + 1, n):
            d = abs(A * self.points[i].x + B * self.points[i].y + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2))
            distance.append(d)

        dmax = max(distance)

        if dmax > self.D:
            swichvalue = True
        else:
            swichvalue = False

        if (not swichvalue):
            for i in range(m + 1, n):
                del self.points[i]
        else:
            for i in range(m + 1, n):
                if (abs(A * self.points[i].x + B * self.points[i].y + C) / math.sqrt(
                        math.pow(A, 2) + math.pow(B, 2)) == dmax):
                    middle = self.points[i]
            self.compress(p1, middle)
            self.compress(middle, p2)

    def printPoint(self):
        """打印数据点"""
        for p in self.points:
            print("%d,%f,%f" % (p.index, p.x, p.y))


def main():
    """测试"""
    # p=Point(20,20,1)
    # print '%d,%d,%d'%(p.x,p.x,p.index)

    d = Douglas()
    d.readPoint()
    # d.printPoint()
    # 结果图形的绘制，抽稀之前绘制
    fig = plt.figure()
    a1 = fig.add_subplot(121)
    dx = []
    dy = []
    for i in range(len(d.points)):
        dx.append(d.points[i].x)
        dy.append(d.points[i].y)
    a1.plot(dx, dy, color='g', linestyle='-')

    d.compress(d.points[0], d.points[len(d.points) - 1])

    # 抽稀之后绘制
    dx1 = []
    dy1 = []
    a2 = fig.add_subplot(122)
    for p in d.points:
        dx1.append(p.x)
        dy1.append(p.y)
    a2.plot(dx1, dy1, color='r', linestyle='-')

    # print "========================\n"
    # d.printPoint()

    plt.show()


if __name__ == '__main__':
    main()