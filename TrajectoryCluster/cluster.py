from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from TrajectoryCluster.aisPoint import AIS
from sqlalchemy import and_
from sqlalchemy import func
from TrajectoryCluster.trajectory import Trajectory
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from TrajectoryCluster.dtw import DTW
import numpy as np
from sklearn.cluster import DBSCAN

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
trajectories = []
trajectory = Trajectory(MMSI)
for data in datas:
    if data.MMSI == MMSI:
        if trajectory.getLength() == 0 or \
                (data.BaseDateTime - trajectory.points[trajectory.getLength() - 1].BaseDateTime).seconds <= 600:
            trajectory.add_point(data)
            continue
    if len(trajectory.points) > 35:
        trajectories.append(trajectory)
    MMSI = data.MMSI
    trajectory = Trajectory(MMSI)
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
    plt.title(title)
    plt.savefig(title, dpi=1080)
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
    trajectory.deleteCircle()
    compressError.append(trajectory.error / trajectory.deleteNum)
print("压缩平均误差：", sum(compressError) / len(compressError))
df = pd.DataFrame(compressError)
df.plot.box(title="Compress Error")
plt.grid(linestyle="--", alpha=0.3)
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

# ===================================调用DTW计算航迹之间的距离======================================
traDistances = []
for i in range(len(trajectories)):
    traDistance = []
    for k in range(i + 1):
        traDistance.append(0)
    for j in range(i + 1, len(trajectories)):
        traDistance.append(DTW(trajectories[i].points, trajectories[j].points))
    traDistances.append(traDistance)
traDistances = np.triu(np.array(traDistances))
traDistances += traDistances.T - np.diag(traDistances.diagonal())

# =============================使用DBSCAN开始聚类=================================================
res = []
for eps in np.arange(0.01, 1, 0.01):
    for min_samples in range(2, 10):
        dbscan = DBSCAN(min_samples=min_samples, eps=eps, leaf_size=1000, metric='precomputed')
        label = dbscan.fit(np.array(traDistances)).labels_
        n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
        # print("聚类个数：", n_clusters)
        # 异常点的个数
        outLiners = np.sum(np.where(dbscan.labels_ == -1, 1, 0))
        # print("异常航迹个数：", outLiners)
        # 统计每个簇的样本个数
        stats = pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values
        res.append(
            {'eps': eps, 'min_samples': min_samples, 'n_clusters': n_clusters, 'outliners': outLiners, 'stats': stats})
# 将迭代后的结果存储到数据框中
df = pd.DataFrame(res)

eps = 0.68
min_samples = 6
dbscan = DBSCAN(min_samples=min_samples, eps=eps, leaf_size=1000, metric='precomputed')
label = dbscan.fit(np.array(traDistances)).labels_
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

# ===========================================绘图聚类效果===============================
colors_dict = {-1: 'red', 0: 'green', 1: 'blue', 2: 'cyan', 3: 'purple', 4: 'magenta', 5: 'darksalmon', 6: 'gray',
               7: 'ivory', 8: 'yellow', 9: 'aqua'}
fig = plt.figure()
a1 = fig.add_subplot(111)
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
    # a1.plot(dx, dy, color=, linestyle='-')
    a1.quiver(dx[:-1], dy[:-1], dx[1:] - dx[:-1], dy[1:] - dy[:-1], scale_units='xy', angles='xy', scale=1,
              color=colorLabel, linestyle='-', width=0.003)
    plt.plot()
plt.title("cluster results")
plt.savefig("cluster results", dpi=1080)
plt.show()

print("结束！")
