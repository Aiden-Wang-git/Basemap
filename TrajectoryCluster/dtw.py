from numpy import array, zeros, argmin, inf, equal, ndim

# =====================设置距离、航向、速度三个变量的权重====================
w1 = 1
w2 = 0
w3 = 0


# 计算序列组成单元之间的距离，可以是欧氏距离，也可以是任何其他定义的距离,这里使用绝对值
def distance(point1, point2):
    d1 = (((point1.LAT - point2.LAT) ** 2 + (point1.LON - point2.LON) ** 2) ** 0.50) / 0.1
    d2 = abs(point1.SOG - point2.SOG) / 15
    d3 = abs(point1.COG - point2.COG) / 360
    return w1 * d1 + w2 * d2 + w3 * d3


# DTW计算序列s1,s2的最小距离
def DTW(s1, s2):
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
    return D1[-1,-1]
# s1 = [1, 3, 2, 4, 2]
# s2 = [0, 3, 4, 2, 2]
#
# print('DTW distance: ', DTW(s1, s2))  # 输出 DTW distance:  2
