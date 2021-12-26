from numpy import array, zeros, argmin, inf, equal, ndim, math
import numpy as np

# =====================设置距离、航向、速度三个变量的权重====================
w1 = 0.5
w2 = 0
w3 = 0.5


# 计算序列组成单元之间的距离，可以是欧氏距离，也可以是任何其他定义的距离,这里使用绝对值
def distanceCom(point1, point2):
    d1 = (((point1.LAT - point2.LAT) ** 2 + (point1.LON - point2.LON) ** 2) ** 0.50) / 0.1
    d2 = abs(point1.SOG - point2.SOG) / 15
    d3 = abs(point1.COG - point2.COG) / 360
    return w1 * d1 + w2 * d2 + w3 * d3

def distance(point1, point2):
    d1 = (((point1.LAT - point2.LAT) ** 2 + (point1.LON - point2.LON) ** 2) ** 0.50) / 0.1
    d2 = abs(point1.SOG - point2.SOG) / 15
    d3 = abs(point1.COG - point2.COG) / 360
    return d1

# def distance(t1, t2):
#     result = 0
#     p1 = t1[0]
#     p2 = t1[1]
#     A = p1.LAT - p2.LAT
#     B = p2.LON - p1.LON
#     C = p1.LON * p2.LAT - p2.LON * p1.LAT
#     result += abs(A * t2[0].LON + B * t2[0].LAT + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2))
#     result += abs(A * t2[1].LON + B * t2[1].LAT + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2))
#     p1 = t2[0]
#     p2 = t2[1]
#     A = p1.LAT - p2.LAT
#     B = p2.LON - p1.LON
#     C = p1.LON * p2.LAT - p2.LON * p1.LAT
#     result += abs(A * t1[0].LON + B * t1[0].LAT + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2))
#     result += abs(A * t1[1].LON + B * t1[1].LAT + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2))
#     return result


# 计算方向距离,根据两个向量，求出它们之间的夹角，返回范围[0,3.14]
def angle(t1, t2, deg=False):
    vec1 = [t1[1].LAT - t1[0].LAT, t1[1].LON - t1[0].LON]
    vec2 = [t2[1].LAT - t2[0].LAT, t2[1].LON - t2[0].LON]
    _angle = np.arctan2(np.abs(np.cross(vec1, vec2)), np.dot(vec1, vec2))
    if deg:
        _angle = np.rad2deg(_angle)
    return _angle


# 计算两条轨迹之间的度量距离
def DTWSpatialDis(s1, s2):
    r, c = len(s1), len(s2)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    # 浅复制
    # print D1
    for i in range(r):
        for j in range(c):
            D1[i, j] = distance(s1[i], s2[j])
    # 生成原始距离矩阵
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j + 1], D0[i + 1, j])
    # 代码核心，动态计算最短距离
    i, j = array(D0.shape) - 2
    # 最短路径
    # print i,j
    p, q = [i], [j]
    count = 1
    while i > 0 or j > 0:
        tb = argmin((D0[i, j], D0[i, j + 1], D0[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
        count += 1
    return D1[-1, -1]/count

# 计算两条轨迹之间的度量距离
def DTWSpatialDisCOM(s1, s2):
    r, c = len(s1), len(s2)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    # 浅复制
    # print D1
    for i in range(r):
        for j in range(c):
            D1[i, j] = distanceCom(s1[i], s2[j])
    # 生成原始距离矩阵
    M = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j + 1], D0[i + 1, j])
    # 代码核心，动态计算最短距离
    i, j = array(D0.shape) - 2
    # 最短路径
    # print i,j
    p, q = [i], [j]
    count = 1
    while i > 0 or j > 0:
        tb = argmin((D0[i, j], D0[i, j + 1], D0[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
        count += 1
    return D1[-1, -1]/count
# DTW计算序列s1,s2的最小距离
def DTW(s1, s2):
    # 如果两条轨迹之间的空间距离相差太多，则直接认为返回一个非常大的值作为这两条轨迹之间的方向距离，目的是为了不让这两条轨迹被聚到同一个类当中
    spatialDsi = DTWSpatialDis(s1, s2)
    if spatialDsi > 0.2:
        return 999
    r, c = len(s1) - 1, len(s2) - 1
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    # 浅复制
    # print D1
    for i in range(r):
        for j in range(c):
            D1[i, j] = angle(s1[i:i + 2], s2[j:j + 2])
    # 生成原始距离矩阵
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j + 1], D0[i + 1, j])
    # 代码核心，动态计算最短距离
    i, j = array(D0.shape) - 2
    # 最短路径
    # print i,j
    p, q = [i], [j]
    count = 1
    while i > 0 or j > 0:
        tb = argmin((D0[i, j], D0[i, j + 1], D0[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
        count += 1
    # return D1[-1, -1] / count
    return D1[-1, -1] / count * (1/(math.e**(-30*(10*spatialDsi-1))+1)+1)


# DTW计算序列s1,s2的最小距离，对比试验
def DTWCompare(s1, s2):
    # 如果两条轨迹之间的空间距离相差太多，则直接认为返回一个非常大的值作为这两条轨迹之间的方向距离，目的是为了不让这两条轨迹被聚到同一个类当中
    r, c = len(s1) - 1, len(s2) - 1
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    # 浅复制
    # print D1
    for i in range(r):
        for j in range(c):
            D1[i, j] = distance(s1[i:i + 2], s2[j:j + 2])
    # 生成原始距离矩阵
    M = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j + 1], D0[i + 1, j])
    # 代码核心，动态计算最短距离
    i, j = array(D0.shape) - 2
    # 最短路径
    # print i,j
    p, q = [i], [j]
    count = 1
    while i > 0 or j > 0:
        tb = argmin((D0[i, j], D0[i, j + 1], D0[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
        count += 1
    return D1[-1, -1] / count


# 对比试验1，使用自定义空间距离
def DTW1(s1, s2):
    r, c = len(s1), len(s2)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    # 浅复制
    # print D1
    for i in range(r):
        for j in range(c):
            D1[i, j] = distance(s1[i], s2[j])
    # 生成原始距离矩阵
    M = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j + 1], D0[i + 1, j])
    # 代码核心，动态计算最短距离
    i, j = array(D0.shape) - 2
    # 最短路径
    # print i,j
    p, q = [i], [j]
    while i > 0 or j > 0:
        tb = argmin((D0[i, j], D0[i, j + 1], D0[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return D1[-1, -1]