from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from TrajectoryCluster.aisPoint import AIS
from sqlalchemy import and_
from sqlalchemy import func
from TrajectoryCluster.trajectory import Trajectory
import time
import matplotlib.pyplot as plt
import pandas as pd
from TrajectoryCluster.my_dtw import DTW, DTWSpatialDis, DTWCompare, DTW1, DTWSpatialDisCOM
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # 计算 轮廓系数，CH 指标，DBI
from TrajectoryCluster.myHausdorff import hausdorff
from pylab import mpl, datetime
from geopy.distance import geodesic
from math import radians, cos, sin, asin, sqrt

from TrajectoryPredict.interpolation import interpolation3
from TrajectoryPredict.mySeq2Seq.mySeq2Seq import *
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
import copy

# ========================全文种使用到的参数==============================

# ===================从数据库中查询数据的条件=============================
top = 33.65
bottom = 33.55
left = -118.30
right = -118.20
begin = '2017-01-01'
end = '2017-02-01'

# ===================航迹预处理的条件=============================
# 一条航迹中至少包含AIS点数目
min_num_in_trajectory = 35
# 相邻AIS点之间的时间阈值,单位S
max_time_between_AIS = 1800
# 插值间隔，如果一条航迹中相邻点时间差大于该值，则插值,单位S
max_interpolation_time = 180
# ===================船舶label和SOG阈值对应关系=====================
max_sog = {
    '1012': 33,  # 客船
    '1010': 19,  # 近海船
    '40': 34,  # 快艇
    '1004': 14,  # 货船
    '1005': 26,  # 工业船
    '1025': 13,  # 拖船
    '1024': 15,  # 油轮
    '1001': 19,  # 渔船
    '1019': 30,  # 游船
    '31': 9,  # 拖船（此类拖船航迹数目较少，整个数据库才14K条数据，可以考虑删除或者和1025合并）
}
# ================DBSCAN聚类时参数===================================
# 豪斯多夫聚类参数
# eps = 0.006
# min_samples = 4
# 混合距离对比试验参数
# eps = 0.16
# min_samples = 6
# 真实实验参数
eps = 0.48
min_samples = 5
# =============================K-means聚类、凝聚聚类参数===============
# n_clusters = 3


# ====================画图时，如遇中文显示问题可加入以下代码============
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# ===========================聚类效果绘图，不同颜色==========================
colors_dict = {-1: 'red', 0: 'green', 1: 'blue', 2: 'cyan', 3: 'purple', 4: 'magenta', 5: 'darksalmon', 6: 'gray',
               7: 'r', 8: 'pink', 9: 'yellow'}


# ==========================================原始航迹提取===============================

# 返回的结果是个字典，key为MMSI，value为按时间排序的AIS点
def getRawTrajectory(top, bottom, left, right, begin, end):
    time1 = datetime.datetime.now()
    print(f"数据库查询范围：({left},{right}),({top},{bottom}),时间范围：{begin}~{end}")
    # 连接数据库
    engine = create_engine("mysql+pymysql://root:123456@localhost:3306/ais?charset=utf8")
    DbSession = sessionmaker(bind=engine)
    session = DbSession()
    datas = session.query(AIS).filter(
        # 取值范围(33.55N,33.65N),(-118.30W,-118.20W),'2017-01-01'~'2018-01-01'
        and_(AIS.LAT >= bottom,
             AIS.LAT <= top,
             AIS.LON >= left,
             AIS.LON <= right,
             AIS.SOG >= 1,
             AIS.BaseDateTime >= begin,
             AIS.BaseDateTime <= end,
             AIS.VesselType.in_(list(max_sog.keys())))
    ).order_by(AIS.BaseDateTime).all()
    session.close()
    trajectories = {}
    for data in datas:
        if trajectories.__contains__(data.MMSI):
            trajectories[data.MMSI].add_point(data)
        else:
            trajectory = Trajectory(data.MMSI, data.VesselType)
            trajectories[data.MMSI] = trajectory
    time2 = datetime.datetime.now()
    print(f"从数据库中提取数据共用时：{time2 - time1}")
    print(f"从数据库中提取轨迹，共有不同类别MMSI:{len(trajectories)}个，AIS点：{len(datas)}个")
    return trajectories


# ========================================第一章：对原始航迹进行分段及删除短航迹================
# 删除AIS点数目小于35个的短航迹，对相邻间隔大于30min的航迹分段
def process1(trajectories):
    trajectories_process1 = {}
    count = 0
    for key in trajectories:
        trajectory = trajectories[key].points
        split_time = [0]
        # 原航迹太短，直接删除
        if len(trajectory) < min_num_in_trajectory:
            continue
        else:
            # 找到分割点
            for i in range(len(trajectory) - 1):
                if (trajectory[i + 1].BaseDateTime - trajectory[i].BaseDateTime).total_seconds() > max_time_between_AIS:
                    split_time.append(i + 1)
            # 往新的字典中添加数据
            if len(split_time) > 1:
                for i in range(len(split_time) - 1):
                    if split_time[i + 1] - split_time[i] >= min_num_in_trajectory:
                        trajectories_process1[key + '-' + str(i)] = trajectory[split_time[i]:split_time[i + 1]]
                        count += len(trajectories_process1[key + '-' + str(i)])
            else:
                trajectories_process1[key] = trajectory
                count += len(trajectories_process1[key])
    print(f"经过process1,轨迹分割之后，共有航迹：{len(trajectories_process1)}条,AIS点：{count}个")
    return trajectories_process1


# 处理位置异常点，以及航速大于35节的点
def process2(trajectories_process1):
    # 用于统计数据中SOG超标数量
    count1 = 0
    # 用于统计根据实际位置SOG超标数量
    count2 = 0
    # 用于记录大于max_interpolation_time的插值数量
    count3 = 0
    # 用于记录需要插值的时间点
    # 用于返回处理后结果
    trajectories_process2 = {}
    # 统计AIS点总数
    count = 0
    for key in trajectories_process1:
        timestamp = []
        trajectory = trajectories_process1[key]
        max_sog_now = max_sog[trajectory[0].VesselType]
        start_time = trajectory[0].BaseDateTime
        # 找到插值的时间戳，从0开始
        for i in range(1, len(trajectory) - 1):
            # 标注超速
            if trajectory[i].SOG > max_sog_now:
                count1 += 1
                timestamp.append((trajectory[i].BaseDateTime - start_time).total_seconds())
            # 实际速度超速
            elif getDistance(trajectory[i], trajectory[i + 1]) / (
                    trajectory[i + 1].BaseDateTime - trajectory[i].BaseDateTime).total_seconds() * 3600 \
                    > 1.852 * max_sog_now \
                    or getDistance(trajectory[i - 1], trajectory[i]) / (
                    trajectory[i].BaseDateTime - trajectory[i - 1].BaseDateTime).total_seconds() * 3600 > \
                    1.852 * max_sog_now:
                count2 += 1
                timestamp.append((trajectory[i].BaseDateTime - start_time).total_seconds())
            # 插值时间间隔
            elif (trajectory[i].BaseDateTime - trajectory[i - 1].BaseDateTime).total_seconds() > max_interpolation_time:
                temp = trajectory[i - 1].BaseDateTime + datetime.timedelta(seconds=max_interpolation_time)
                while temp < trajectory[i].BaseDateTime:
                    timestamp.append((temp - start_time).total_seconds())
                    count3 += 1
                    temp = temp + datetime.timedelta(seconds=max_interpolation_time)
        i = len(trajectory) - 1
        temp = trajectory[i - 1].BaseDateTime + datetime.timedelta(seconds=max_interpolation_time)
        while temp < trajectory[i].BaseDateTime:
            timestamp.append((temp - start_time).total_seconds())
            count3 += 1
            temp = temp + datetime.timedelta(seconds=max_interpolation_time)
        if len(timestamp) > 0:
            trajectory = interpolation3(trajectory=trajectory, inter_time=timestamp)
        trajectories_process2[key] = trajectory
        count += len(trajectory)
    print(f"process2时,AIS插值点count1:{count1}个，count2:{count2}个，count3:{count3}个")
    print(f"process2异常点剔除及插值之后,共有航迹：{len(trajectories_process2)}条,AIS点：{count}个")
    for key in trajectories_process2:
        trajectory = Trajectory(trajectories_process2[key][0].MMSI, trajectories_process2[key][0].VesselType)
        trajectory.set_points(trajectories_process2[key])
        trajectories_process2[key] = trajectory
    return trajectories_process2, count


# ========================================第二章：对原始航迹进行分段及删除短航迹================

# 改进后D-P压缩
def trajectory_compress(trajectories, count):
    trajectories_process3 = trajectories
    compressError = []
    for key in trajectories_process3:
        trajectory = trajectories_process3[key]
        trajectory.compress(trajectory.points[0], trajectory.points[trajectory.count - 1])
        # D-P压缩改进具体步骤
        trajectory.deleteCircle()
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
    for key in trajectories_process3:
        aisNumAfter += len(trajectories_process3[key].points)
    print("压缩后共有AIS点：", aisNumAfter)
    print("压缩率为：", 1 - aisNumAfter / count)
    return trajectories_process3


# ========================================第三章：对航迹完成聚类================
# 1.计算航迹之间的度量距离
def get_DBLD(trajectories_process3):
    trajectories = []
    for key in trajectories_process3:
        trajectories.append(trajectories_process3[key])
    # 保存航迹距离
    tra_distances_cluster = []
    # 保存位置距离
    tra_distances_SC = []
    # 统计计算次数
    countNum = 0
    time1 = datetime.datetime.now()
    for i in range(len(trajectories)):
        traDistance = []
        traDistanceSpa = []
        for k in range(i + 1):
            traDistance.append(0)
            traDistanceSpa.append(0)
        for j in range(i + 1, len(trajectories)):
            # 本文实验
            traDistance.append(DTW(trajectories[i].points, trajectories[j].points))
            # 混合距离实验
            # traDistance.append(DTWSpatialDisCOM(trajectories[i].points, trajectories[j].points))
            # 豪斯多夫距离对比试验
            # traDistance.append(hausdorff(trajectories[i].points, trajectories[j].points))
            # 计算SC得分需要的度量距离
            traDistanceSpa.append(DTWSpatialDis(trajectories[i].points, trajectories[j].points))
            countNum = countNum + len(trajectories[i].points) * len(trajectories[j].points)
        tra_distances_cluster.append(traDistance)
        tra_distances_SC.append(traDistanceSpa)
    tra_distances_cluster = np.triu(np.array(tra_distances_cluster))
    tra_distances_SC = np.triu(np.array(tra_distances_SC))
    tra_distances_cluster += tra_distances_cluster.T - np.diag(tra_distances_cluster.diagonal())
    tra_distances_SC += tra_distances_SC.T - np.diag(tra_distances_SC.diagonal())
    time2 = datetime.datetime.now()
    print(f"计算航迹间度量距离时比较次数：{countNum},用时：{time2 - time1}")
    return tra_distances_cluster, tra_distances_SC


# 2.尝试聚类算法的参数
# 凝聚聚类尝试不同参数
def get_agg_cluster_parameters(tra_distances_cluster, tra_distances_SC):
    res = []
    time1 = datetime.datetime.now()
    for k in np.arange(2, 20, 1):
        agg_model = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
        label = agg_model.fit(np.array(tra_distances_cluster)).labels_
        try:
            score = silhouette_score(np.array(tra_distances_SC), label, metric='precomputed')
        except ValueError:
            score = -1
        n_clusters = len([i for i in set(agg_model.labels_) if i != -1])
        # print("聚类个数：", n_clusters)
        # 异常点的个数
        outLiners = np.sum(np.where(agg_model.labels_ == -1, 1, 0))
        # print("异常航迹个数：", outLiners)
        # 统计每个簇的样本个数
        stats = pd.Series([i for i in agg_model.labels_ if i != -1]).value_counts().values
        score = -1
        if stats.size > 2:
            traDistancesSC = []
            labelSC = []
            for i in range(len(label)):
                line = []
                if label[i] == -1:
                    continue
                for j in range(len(label)):
                    if not label[i] == -1 and not label[j] == -1:
                        line.append(tra_distances_cluster[i][j])
                traDistancesSC.append(line)
            for i in range(len(label)):
                if not label[i] == -1:
                    labelSC.append(label[i])
            score = silhouette_score(np.array(traDistancesSC), labelSC, metric='precomputed')
        res.append(
            {'eps': eps, 'min_samples': min_samples, 'n_clusters': n_clusters, 'outliners': outLiners,
             'stats': stats,
             'score': score})
    time2 = datetime.datetime.now()
    print(f"获取聚类参数用时：{time2 - time1}")
    # 将迭代后的结果存储到数据框中
    df = pd.DataFrame(res)
    return df.sort_values(by=['score'], ascending=False)


# K-means聚类尝试获取参数
def get_Kmeans_cluster_parameters(tra_distances_cluster, tra_distances_SC):
    res = []
    time1 = datetime.datetime.now()
    for k in np.arange(2, 20, 1):
        kmeans_model = KMeans(n_clusters=k, random_state=1, precompute_distances='precomputed')
        label = kmeans_model.fit(np.array(tra_distances_cluster)).labels_
        try:
            score = silhouette_score(np.array(tra_distances_SC), label, metric='precomputed')
        except ValueError:
            score = -1
        n_clusters = len([i for i in set(kmeans_model.labels_) if i != -1])
        # print("聚类个数：", n_clusters)
        # 异常点的个数
        outLiners = np.sum(np.where(kmeans_model.labels_ == -1, 1, 0))
        # print("异常航迹个数：", outLiners)
        # 统计每个簇的样本个数
        stats = pd.Series([i for i in kmeans_model.labels_ if i != -1]).value_counts().values
        score = -1
        if stats.size > 2:
            traDistancesSC = []
            labelSC = []
            for i in range(len(label)):
                line = []
                if label[i] == -1:
                    continue
                for j in range(len(label)):
                    if not label[i] == -1 and not label[j] == -1:
                        line.append(tra_distances_cluster[i][j])
                traDistancesSC.append(line)
            for i in range(len(label)):
                if not label[i] == -1:
                    labelSC.append(label[i])
            score = silhouette_score(np.array(traDistancesSC), labelSC, metric='precomputed')
        res.append(
            {'eps': eps, 'min_samples': min_samples, 'n_clusters': n_clusters, 'outliners': outLiners,
             'stats': stats,
             'score': score})
    time2 = datetime.datetime.now()
    print(f"获取聚类参数用时：{time2 - time1}")
    # 将迭代后的结果存储到数据框中
    df = pd.DataFrame(res)
    return df.sort_values(by=['score'], ascending=False)


# DBSCAN尝试聚类算法的参数
def get_DBSCAN_cluster_parameters(tra_distances_cluster, tra_distances_SC):
    res = []
    time1 = datetime.datetime.now()
    for eps in np.arange(0.01, 1, 0.01):
        for min_samples in range(2, 10):
            dbscan = DBSCAN(min_samples=min_samples, eps=eps, leaf_size=1000, metric='precomputed')
            label = dbscan.fit(np.array(tra_distances_cluster)).labels_
            try:
                score = silhouette_score(np.array(tra_distances_SC), label, metric='precomputed')
            except ValueError:
                score = -1
            n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
            # print("聚类个数：", n_clusters)
            # 异常点的个数
            outLiners = np.sum(np.where(dbscan.labels_ == -1, 1, 0))
            # print("异常航迹个数：", outLiners)
            # 统计每个簇的样本个数
            stats = pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values
            score = -1
            if stats.size > 2:
                traDistancesSC = []
                labelSC = []
                for i in range(len(label)):
                    line = []
                    if label[i] == -1:
                        continue
                    for j in range(len(label)):
                        if not label[i] == -1 and not label[j] == -1:
                            line.append(tra_distances_cluster[i][j])
                    traDistancesSC.append(line)
                for i in range(len(label)):
                    if not label[i] == -1:
                        labelSC.append(label[i])
                score = silhouette_score(np.array(traDistancesSC), labelSC, metric='precomputed')
            res.append(
                {'eps': eps, 'min_samples': min_samples, 'n_clusters': n_clusters, 'outliners': outLiners,
                 'stats': stats,
                 'score': score})
    time2 = datetime.datetime.now()
    print(f"获取聚类参数用时：{time2 - time1}")
    # 将迭代后的结果存储到数据框中
    df = pd.DataFrame(res)
    return df.sort_values(by=['score'], ascending=False)


# 3.聚类结果统计
def cluster_result(tra_distances_SC, trajectories_process3, label, trajectories_process2_copy):
    score = silhouette_score(np.array(tra_distances_SC), label, metric='precomputed')
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
                line.append(tra_distances_SC[i][j])
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
    i = 0
    for key in trajectories_process3:
        trajectories_process3[key].label = label[i]
        trajectories_process2_copy[key].label = label[i]
        i += 1
    return n_clusters


# =========================================工具函数====================================

# 1.根据两个AIS点，计算两者之间的实际距离
def getDistance(pointA, pointB):
    # 这个方法太慢
    # return geodesic((pointA.LAT, pointA.LON), (pointB.LAT, pointB.LON)).km
    lng1 = pointA.LON
    lat1 = pointA.LAT
    lng2 = pointB.LON
    lat2 = pointB.LAT
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 3)
    return distance


# 2.构造Seq2Seq训练模型model, 以及进行新序列预测时需要的的Encoder模型:encoder_model 与Decoder模型:decoder_model
def define_models(n_input, n_output, n_units):
    # 训练模型中的encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]  # 仅保留编码状态向量
    # 训练模型中的decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # 新序列预测时需要的encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # 新序列预测时需要的decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # 返回需要的三个模型
    return model, encoder_model, decoder_model


# 3.航迹展示
def drawTrajectory(title, trajectories):
    fig = plt.figure()
    a1 = fig.add_subplot(111)
    for key in trajectories:
        trajectory = trajectories[key]
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


# 4.聚类结果绘图展示
def draw_cluster_result(trajectories_process3):
    trajectories = []
    for key in trajectories_process3:
        trajectories.append(trajectories_process3[key])
    fig = plt.figure()
    a1 = fig.add_subplot(111)
    colorLegend = []
    colorIndex = 1
    a1 = fig.add_subplot(111)
    a1.set_ylim(bottom=33.55)
    a1.set_ylim(top=33.65)
    a1.set_xlim(left=-118.30)
    a1.set_xlim(right=-118.20)
    trajectories.sort(key=lambda trajectory: trajectory.label)
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


# ==================================主函数部分==========================================

# =============================第一章：航迹数据的预处理===================================
print("========================第一章：航迹数据的预处理================================")
trajectories = getRawTrajectory(top=top, bottom=bottom, left=left, right=right, begin=begin, end=end)
trajectories_process1 = process1(trajectories=trajectories)
trajectories_process2, count = process2(trajectories_process1=trajectories_process1)
drawTrajectory("第一章提取到的航迹数据", trajectories_process2)
# =============================第二章：航迹的压缩===================================
print("========================第二章：航迹的压缩================================")
# 首先把原数据复制一份，用于训练模型
trajectories_process2_copy = copy.deepcopy(trajectories_process2)
trajectories_process3 = trajectory_compress(trajectories=trajectories_process2, count=count)
drawTrajectory("第二章压缩后的航迹数据", trajectories_process3)
# =============================第三章：航迹的聚类===================================
print("========================第三章：航迹的聚类================================")
tra_distances_cluster, tra_distances_SC = get_DBLD(trajectories_process3=trajectories_process3)
# ========DBSCAN聚类=============
print(f"这是DBSCAN聚类,eps={eps},min_samples={min_samples}")
# df = get_DBSCAN_cluster_parameters(tra_distances_cluster=tra_distances_cluster, tra_distances_SC=tra_distances_SC)
dbscan = DBSCAN(min_samples=min_samples, eps=eps, leaf_size=1000, metric='precomputed')
labels = dbscan.fit(np.array(tra_distances_cluster)).labels_
# ===========Agg聚类=============
# print(f"这是Agg聚类,n_clusters={n_clusters}")
# df = get_agg_cluster_parameters(tra_distances_cluster=tra_distances_cluster, tra_distances_SC=tra_distances_SC)
# agg_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
# label = agg_model.fit(np.array(tra_distances_cluster)).labels_
# K-means聚类
# print(f"这是K-means聚类,n_clusters={n_clusters}")
# df = get_Kmeans_cluster_parameters(tra_distances_cluster=tra_distances_cluster, tra_distances_SC=tra_distances_SC)
# kmeans_model = KMeans(n_clusters=n_clusters, random_state=1, precompute_distances='precomputed')
# label = kmeans_model.fit(np.array(tra_distances_cluster)).labels_
cluster_num = cluster_result(tra_distances_SC=tra_distances_SC, trajectories_process3=trajectories_process3,
                             label=labels, trajectories_process2_copy=trajectories_process2_copy)
draw_cluster_result(trajectories_process3=trajectories_process3)
# =============================第四章：seq2seq预测=======================================
print("============================第四章：seq2seq预测======================================")
for label in range(0, cluster_num):
    trajectories_seq2seq = {}
    for key in trajectories_process2_copy:
        if trajectories_process2_copy[key].label == label:
            trajectories_seq2seq[key] = trajectories_process2_copy[key].points
    data_to_seq2seq(trajectories_seq2seq)
    print(f"label{label}模型训练完成")

# 训练模型
# train_seq2seq_model()
# 预测用例船舶MMSI
# test1 = '367580930-24'
# test2 = '367724740-0'
# test3 = '367719640-4'
# test4 = '367719640-31'
# model_predict(trajectories_process2[test1])
# model_predict(trajectories_process2[test2])
# model_predict(trajectories_process2[test3])
# model_predict(trajectories_process2[test4])
