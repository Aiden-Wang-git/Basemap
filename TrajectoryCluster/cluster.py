from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from TrajectoryCluster.aisPoint import AIS
from sqlalchemy import and_
from sqlalchemy import func
from TrajectoryCluster.trajectory import Trajectory
import time
import matplotlib.pyplot as plt
import pandas as pd
from TrajectoryCluster.dtw import DTW, DTWSpatialDis, DTWCompare, DTW1, DTWSpatialDisCOM
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # 计算 轮廓系数，CH 指标，DBI
from TrajectoryCluster.myHausdorff import hausdorff
# 如遇中文显示问题可加入以下代码
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

engine = create_engine("mysql+pymysql://root:123456@localhost:3306/ais?charset=utf8")
# 创建session
DbSession = sessionmaker(bind=engine)
session = DbSession()
# 测试往数据库中写入数据
# test_ais = AIS(1,"王军号MMSI","1996-11-27",1.0,1.0,1.0,1.0,1.0,1.0)
# session.add(test_ais)
# session.commit()
# 测试从数据库中读取数据
# testAIS = session.query(AIS).filter(AIS.MMSI=='王军号MMSI').one()
# print("type:",type(testAIS))
# print("生日",testAIS.BaseDateTime)



# ================================航迹提取=============================
# 读取研究范围内所有航速大于1的AIS点
datas = session.query(AIS).filter(
    and_(AIS.LAT >= 33.55,
         AIS.LAT <= 33.65,
         AIS.LON >= -118.30,
         AIS.LON <= -118.20,
         AIS.SOG >= 1,
         AIS.BaseDateTime >= '2017-01-01',
         AIS.BaseDateTime <= '2017-02-01')).order_by(AIS.MMSI, AIS.BaseDateTime).all()
session.close()
# 将航迹依据MMSI分开
# 将间隔时间大于10min的轨迹断开
MMSI = datas[0].MMSI
vesselType = datas[0].VesselType
trajectories = []
trajectory = Trajectory(MMSI, vesselType)
for data in datas:
    if data.MMSI == MMSI:
        if trajectory.getLength() == 0 or \
                (data.BaseDateTime - trajectory.points[trajectory.getLength() - 1].BaseDateTime).seconds <= 600:
            trajectory.add_point(data)
            continue
    if len(trajectory.points) > 35:
        trajectories.append(trajectory)
    MMSI = data.MMSI
    vesselType = data.VesselType
    trajectory = Trajectory(MMSI, vesselType)
    trajectory.add_point(data)
print("共有轨迹条数：", len(trajectories))


# ===============================航迹展示===================================
def drawTrajectory(title):
    global trajectory
    fig = plt.figure()
    a1 = fig.add_subplot(111)
    for trajectory in trajectories:
        dx = []
        dy = []
        for i in range(trajectory.getLength()):
            dx.append(trajectory.points[i].LON)
            dy.append(trajectory.points[i].LAT)
        dx = np.array(dx)
        dy = np.array(dy)
        a1.quiver(dx[:-1], dy[:-1], dx[1:] - dx[:-1], dy[1:] - dy[:-1], scale_units='xy', angles='xy', scale=1,
                  color='r', linestyle='-', width=0.003)
    plt.xlabel('经度/°')
    plt.ylabel('纬度/°')
    # plt.title(title)
    plt.savefig(title, dpi=1080, bbox_inches='tight')
    plt.show()


# 航迹压缩前画图
drawTrajectory("Before Compress")

# 航迹压缩前共存在AIS点个数
aisNumBefore = 0
for trajectory in trajectories:
    aisNumBefore += len(trajectory.points)
print("压缩前共有AIS点：", aisNumBefore)

# =======================================航迹压缩==================================
compressError = []
for trajectory in trajectories:
    trajectory.compress(trajectory.points[0], trajectory.points[trajectory.count - 1])
    # trajectory.deleteCircle()
    compressError.append(trajectory.error / trajectory.deleteNum)
print("压缩平均误差：", sum(compressError) / len(compressError))
df = pd.DataFrame(compressError)
df.plot.box(title="Compress Error")
plt.grid(linestyle="--", alpha=0.1)
plt.xlabel('经度/°')
plt.ylabel('纬度/°')
plt.show()

# 航迹压缩后共存在AIS点个数
aisNumAfter = 0
for trajectory in trajectories:
    aisNumAfter += len(trajectory.points)
print("压缩后共有AIS点：", aisNumAfter)
print("压缩率为：", 1 - aisNumAfter / aisNumBefore)

# 航迹压缩后画图
drawTrajectory("After Compress")

# print(len(datas))

# ================================程序开始时间=========================
startTime = time.time()

# ===================================调用DTW计算航迹之间的距离======================================
# 保存航迹距离
traDistances = []
# 保存位置距离
traDistancesSpa = []
# 统计计算次数
countNum = 0
for i in range(len(trajectories)):
    traDistance = []
    traDistanceSpa = []
    for k in range(i + 1):
        traDistance.append(0)
        traDistanceSpa.append(0)
    for j in range(i + 1, len(trajectories)):
        # 本文实验
        # traDistance.append(DTW(trajectories[i].points, trajectories[j].points))
        # 混合距离实验
        traDistance.append(DTWSpatialDisCOM(trajectories[i].points, trajectories[j].points))
        # 豪斯多夫距离对比试验
        # traDistance.append(hausdorff(trajectories[i].points, trajectories[j].points))
        # 计算SC得分需要的度量距离
        traDistanceSpa.append(DTWSpatialDis(trajectories[i].points, trajectories[j].points))
        countNum = countNum + len(trajectories[i].points) * len(trajectories[j].points)
    traDistances.append(traDistance)
    traDistancesSpa.append(traDistanceSpa)
traDistances = np.triu(np.array(traDistances))
traDistancesSpa = np.triu(np.array(traDistancesSpa))
traDistances += traDistances.T - np.diag(traDistances.diagonal())
traDistancesSpa += traDistancesSpa.T - np.diag(traDistancesSpa.diagonal())
print("计算DTW距离时比较次数：" + str(countNum))



# =========================测试聚类时参数===================
# res = []
# for eps in np.arange(0.01, 1, 0.01):
#     for min_samples in range(2, 10):
#         dbscan = DBSCAN(min_samples=min_samples, eps=eps, leaf_size=1000, metric='precomputed')
#         label = dbscan.fit(np.array(traDistances)).labels_
#         try:
#             score = silhouette_score(np.array(traDistancesSpa), label, metric='precomputed')
#         except ValueError:
#             score = -1
#         n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
#         # print("聚类个数：", n_clusters)
#         # 异常点的个数
#         outLiners = np.sum(np.where(dbscan.labels_ == -1, 1, 0))
#         # print("异常航迹个数：", outLiners)
#         # 统计每个簇的样本个数
#         stats = pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values
#         score = -1
#         if stats.size > 2:
#             traDistancesSC = []
#             labelSC = []
#             for i in range(len(label)):
#                 line = []
#                 if label[i] == -1:
#                     continue
#                 for j in range(len(label)):
#                     if not label[i] == -1 and not label[j] == -1:
#                         line.append(traDistances[i][j])
#                 traDistancesSC.append(line)
#             for i in range(len(label)):
#                 if not label[i] == -1:
#                     labelSC.append(label[i])
#             score = silhouette_score(np.array(traDistancesSC), labelSC, metric='precomputed')
#         res.append(
#             {'eps': eps, 'min_samples': min_samples, 'n_clusters': n_clusters, 'outliners': outLiners, 'stats': stats,
#              'score': score})
# # 将迭代后的结果存储到数据框中
# df = pd.DataFrame(res)


# =============================使用DBSCAN开始聚类=================================================
# 豪斯多夫聚类参数
# eps = 0.006
# min_samples = 4
# 混合距离对比试验参数
eps = 0.16
min_samples = 6
# 真实实验参数
# eps = 0.27
# min_samples = 5
dbscan = DBSCAN(min_samples=min_samples, eps=eps, leaf_size=1000, metric='precomputed')
label = dbscan.fit(np.array(traDistances)).labels_
# 评价聚类的效果

# =====================================程序结束时间==========================
endTime = time.time()


score = silhouette_score(np.array(traDistancesSpa), label, metric='precomputed')
print("聚类效果SC得分：", score)
# 去除异常样本的SC得分
traDistancesSC = []
labelSC = []
for i in range(len(label)):
    line = []
    if label[i] == -1:
        continue
    for j in range(len(label)):
        if not label[i] == -1 and not label[j] == -1:
            line.append(traDistancesSpa[i][j])
    traDistancesSC.append(line)
for i in range(len(label)):
    if not label[i] == -1:
        labelSC.append(label[i])
score = silhouette_score(np.array(traDistancesSC), labelSC, metric='precomputed')
print("去除异常样本聚类效果SC得分：", score)
n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
print("聚类个数：", n_clusters)
# 异常点的个数
outLiners = np.sum(np.where(dbscan.labels_ == -1, 1, 0))
print("异常航迹个数：", outLiners)
# 统计每个簇的样本个数
stats = pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values
for i in range(len(stats)):
    print("类别", i, "共有航迹数量：", stats[i])
# 给航迹打上类别标签
for i in range(len(trajectories)):
    trajectories[i].label = label[i]

# ===================展示不同label船舶的vesselType===================
for i in range(n_clusters):
    print("label", i, "的船舶type:")
    for trajectory in trajectories:
        if trajectory.label == i:
            print(trajectory.vesselType, end=",")
    print()

# ===========================================绘图聚类效果===============================
colors_dict = {-1: 'red', 0: 'green', 1: 'blue', 2: 'cyan', 3: 'purple', 4: 'magenta', 5: 'darksalmon', 6: 'gray',
               7: 'r', 8: 'pink', 9: 'yellow'}
fig = plt.figure()
a1 = fig.add_subplot(111)
colorLegend = []
colorIndex = 1
a1 = fig.add_subplot(111)
a1.set_ylim(bottom=33.55)
a1.set_ylim(top=33.65)
a1.set_xlim(left=-118.30)
a1.set_xlim(right=-118.20)
for trajectory in trajectories:
    dx = []
    dy = []
    colorLabel = colors_dict[trajectory.label]
    if -1 == trajectory.label:
        continue
    for point in trajectory.points:
        dx.append(point.LON)
        dy.append(point.LAT)
    dx = np.array(dx)
    dy = np.array(dy)
    a1.plot(dx, dy, color=colorLabel, linestyle='-')
    if colorLegend.__contains__(trajectory.label):
        a1.quiver(dx[:-1], dy[:-1], dx[1:] - dx[:-1], dy[1:] - dy[:-1], scale_units='xy', angles='xy', scale=1,
                  color=colorLabel, linestyle='-', width=0.003)
    else:
        a1.quiver(dx[:-1], dy[:-1], dx[1:] - dx[:-1], dy[1:] - dy[:-1], scale_units='xy', angles='xy', scale=1,
                  color=colorLabel, linestyle='-', width=0.003, label="label" + str(colorIndex))
        colorIndex = colorIndex + 1
        plt.legend(loc=4)
        colorLegend.append(trajectory.label)
    plt.plot()
plt.xlabel('经度/°')
plt.ylabel('纬度/°')
# plt.title("label")
plt.savefig("cluster results", dpi=1080, bbox_inches='tight')
plt.show()

# 分别展示不同label的航迹
# for aaa in range(len(colorLegend)):
#     fig = plt.figure()
#     a1 = fig.add_subplot(111)
#     a1.set_ylim(bottom=33.55)
#     a1.set_ylim(top=33.65)
#     a1.set_xlim(left=-118.30)
#     a1.set_xlim(right=-118.20)
#     for trajectory in trajectories:
#         dx = []
#         dy = []
#         colorLabel = colors_dict[trajectory.label]
#         if aaa != trajectory.label:
#             continue
#         for point in trajectory.points:
#             dx.append(point.LON)
#             dy.append(point.LAT)
#         dx = np.array(dx)
#         dy = np.array(dy)
#         # a1.plot(dx, dy, color=, linestyle='-')
#         if colorLegend.__contains__(trajectory.label):
#             a1.quiver(dx[:-1], dy[:-1], dx[1:] - dx[:-1], dy[1:] - dy[:-1], scale_units='xy', angles='xy', scale=1,
#                       color=colorLabel, linestyle='-', width=0.003)
#         plt.plot()
#     plt.xlabel('经度/°')
#     plt.ylabel('纬度/°')
#     plt.title("label"+str(aaa+1))
#     plt.savefig("cluster results"+str(aaa), dpi=1080, bbox_inches='tight')
#     print("label"+str(aaa)+"绘图完成。。。。。")
#     # plt.show()


print("结束！")
print("共用时",str(endTime-startTime))
