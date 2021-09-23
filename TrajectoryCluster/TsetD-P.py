from geog import distance
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


class Douglas_Peuker:
    "Douglas–Peucker algorithm压缩轨迹"

    def __init__(self, traj, dMax=1.5):

        print('原始轨迹长度：', len(traj))
        plt.figure()  # 画图
        plt.subplot(1, 2, 1)  # 画图
        plt.plot(np.array(traj)[:, 0], np.array(traj)[:, 1], 'r-')
        plt.title('init_traj')
        traj = [[i, x[0], x[1]] for i, x in enumerate(traj)]
        douglasPecker_traj = self.douglasPecker(traj, dMax)
        print('压缩后轨迹长度：', len(douglasPecker_traj))
        plt.subplot(1, 2, 2)  # 画图
        plt.plot(np.array(douglasPecker_traj)[:, 0], np.array(douglasPecker_traj)[:, 1], 'b-')
        plt.title('douglasPecker_traj')
        plt.show()
        plt.close()

    def douglasPecker(self, coordinate, dMax):
        """

        :param coordinate: 原始轨迹
        :param dMax: 允许最大距离误差
        :return: douglasResult 抽稀后的轨迹
        """
        result = self.compressLine(coordinate, [], 0, len(coordinate) - 1, dMax)
        result.append(coordinate[0])
        result.append(coordinate[-1])
        result = pd.DataFrame(result, columns=['myIndex', 'lat', 'lng'])
        result.sort_values(by=['myIndex'], inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result.loc[:, ['lat', 'lng']].values.tolist()

    def compressLine(self, coordinate, result, start, end, dMax):
        "递归方式压缩轨迹"
        if start < end:
            maxDist = 0
            currentIndex = 0
            startPoint = coordinate[start][1:]
            endPoint = coordinate[end][1:]
            for i in range(start + 1, end):
                currentDist = self.disToSegment(startPoint, endPoint, coordinate[i][1:])
                if currentDist > maxDist:
                    maxDist = currentDist
                    currentIndex = i
            if maxDist >= dMax:
                # 将当前点加入到过滤数组中
                result.append(coordinate[currentIndex])
                # 将原来的线段以当前点为中心拆成两段，分别进行递归处理
                self.compressLine(coordinate, result, start, currentIndex, dMax)
                self.compressLine(coordinate, result, currentIndex, end, dMax)
        return result

    def disToSegment(self, start, end, center):
        "计算垂距，用海伦公式计算面积"
        a = distance(start, end)
        b = distance(start, center)
        c = distance(end, center)
        p = (a + b + c) / 2
        s = np.sqrt(abs(p * (p - a) * (p - b) * (p - c)))
        return s * 2 / a


traj = [[0,1],[1,1.5],[2,0.5],[3,1],[4,1.2],[5,0.8],[6,3],[5,1],[6,0],[7,-1],[8,-2],[9,-3]]

if __name__ == '__main__':
    "traj是一个二维数组格式的经纬度列表,lng在前，lat在后"
    Douglas_Peuker(traj, dMax=5)