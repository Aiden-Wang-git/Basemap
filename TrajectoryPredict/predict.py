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

# 画图时，如遇中文显示问题可加入以下代码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 连接数据库
engine = create_engine("mysql+pymysql://root:123456@localhost:3306/ais?charset=utf8")
DbSession = sessionmaker(bind=engine)
session = DbSession()

# ==========================================航迹提取===============================
