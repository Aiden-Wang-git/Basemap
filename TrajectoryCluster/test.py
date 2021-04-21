import numpy as np
import matplotlib.pyplot as plt

# X = np.random.randn(1, 10)
# Y = np.random.randn(1, 10)
# print(X,Y)
# label = np.array([1,1,0,0,0,0,0,1,1,1])
# # 按照类比生成颜色
# # colors = plt.cm.Spectral(label)
# plt.scatter(X.reshape(10), Y.reshape(10), c =label, s = 180, cmap = plt.cm.Spectral)
# plt.show()

X = []
Y = []
# for i in range(10):
#     X.append(i)
#     Y.append(2*i)
X.append(0)
Y.append(0)
X.append(9)
Y.append(18)
X.append(11)
Y.append(10)
X.append(12)
Y.append(9)
X.append(13)
Y.append(11)
X.append(14)
Y.append(11)
X.append(25)
Y.append(22)
# for i in range(14,25):
#     X.append(i)
#     Y.append(i-3)
plt.plot(X, Y)
plt.title("before")
plt.show()


def deleteCircle(X, Y):
    #  用于记录下标
    flag = True
    while (flag):
        flag = False
        distance = getDistance(X, Y)
        for i in range(1, len(distance) - 1):
            if distance[i] / distance[i - 1] < 0.3 or distance[i] / distance[i + 1] < 0.3:
                lat = (X[i + 1] + X[i]) / 2
                lon = (Y[i + 1] + Y[i]) / 2
                X.pop(i)
                X.pop(i)
                Y.pop(i)
                Y.pop(i)
                X.insert(i, lat)
                Y.insert(i, lon)
                flag = True
                break


def getDistance(X, Y):
    distance = []
    for i in range(1, len(X)):
        distance.append(((X[i] - X[i - 1]) ** 2 + (Y[i] - Y[i - 1]) ** 2) ** 0.5)
    return distance


deleteCircle(X, Y)
plt.plot(X, Y)
plt.title("after")
plt.show()
