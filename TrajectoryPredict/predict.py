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
from pylab import mpl
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

# ====================画图时，如遇中文显示问题可加入以下代码============
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# ==========================================原始航迹提取===============================
# 返回的结果是个字典，key为MMSI，value为按时间排序的AIS点
def getRawTrajectory(top, bottom, left, right, begin, end):
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
            trajectories[data.MMSI].append(data)
        else:
            trajectories[data.MMSI] = [data]
    print(f"从数据库中提取轨迹，共有不同类别MMSI:{len(trajectories)}个，AIS点：{len(datas)}个")
    return trajectories


# ==========================================对原始航迹进行分段及删除短航迹================
# 删除AIS点数目小于35个的短航迹，对相邻间隔大于30min的航迹分段
def process1(trajectories):
    trajectories_process1 = {}
    count = 0
    for key in trajectories:
        trajectory = trajectories[key]
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
    print(f"经过process1之后，共有航迹：{len(trajectories_process1)}条,AIS点：{count}个")
    return trajectories_process1


# 处理位置异常点，以及航速大于35节的点
def process2(trajectories_process1):
    # 用于统计数据中SOG超标数量
    count1 = 0
    # 用于统计根据实际位置SOG超标数量
    count2 = 0
    # 用于记录需要插值的时间点
    for key in trajectories_process1:
        timestamp = []
        trajectory = trajectories_process1[key]
        max_sog_now = max_sog[trajectory[0].VesselType]
        start_time = trajectory[0].BaseDateTime
        for i in range(1, len(trajectory) - 1):
            if trajectory[i].SOG > max_sog_now:
                count1 += 1
                timestamp.append((trajectory[i].BaseDateTime - start_time).total_seconds())
            elif getDistance(trajectory[i], trajectory[i + 1]) / (
                    trajectory[i + 1].BaseDateTime - trajectory[i].BaseDateTime).total_seconds() * 3600 > 1.852 * max_sog_now \
                    or getDistance(trajectory[i - 1], trajectory[i]) / (
                    trajectory[i].BaseDateTime - trajectory[i - 1].BaseDateTime).total_seconds() * 3600 > 1.852 * max_sog_now:
                count2 += 1
                timestamp.append((trajectory[i].BaseDateTime - start_time).total_seconds())
        if len(timestamp)>0:
            interpolation3(trajectory=trajectory, inter_time=timestamp)
    print(f"经过process2之后,AIS点超速count1:{count1}个")
    print(f"经过process2之后,AIS点异常count2:{count2}个")
    return trajectories_process1


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


# 构造Seq2Seq训练模型model, 以及进行新序列预测时需要的的Encoder模型:encoder_model 与Decoder模型:decoder_model
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


# ==================================主函数部分==========================================

# =============================第一章：航迹数据的预处理===================================
trajectories = getRawTrajectory(top=top, bottom=bottom, left=left, right=right, begin=begin, end=end)
trajectories_process1 = process1(trajectories=trajectories)
trajectories_process2 = process2(trajectories_process1=trajectories_process1)

# =============================第四章：seq2seq预测=======================================
data_to_seq2seq(trajectories_process2)
# 训练模型
# train_seq2seq_model()
# 预测用例船舶MMSI
test1 = '367580930-24'
test2 = '367724740-0'
test3 = '367719640-4'
test4 = '367719640-31'
model_predict(trajectories_process2[test1])
model_predict(trajectories_process2[test2])
model_predict(trajectories_process2[test3])
model_predict(trajectories_process2[test4])
