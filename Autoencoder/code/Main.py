import copy
import heapq
import os
import random

import pymysql as pymysql
from scipy.interpolate import interp1d
from nltk.cluster.kmeans import KMeansClusterer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import DTW
from threading import Thread
pymysql.install_as_MySQLdb()
import MySQLdb
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from math import radians, cos, sin, asin, sqrt
import numpy as np
from geopy.distance import geodesic
import time, datetime
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from itertools import chain
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import math

# 预测船舶主键ID,船舶MMSI，预测点时间
ID = "38511417"
GMMNum = 9  # GMM聚类个数
MMSISelect = ""  # 存放目标船舶MMSI
BaseDateTimeSelect = ""  # 存放初始点时间
# epsNum = 0.1
# min_samplesNum = 10
dirs = '../results/{0}'.format(ID)  # 结果文件存放位置

class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

# 读取预测船舶的信息，并将起点角度转换为-180~180，返回真实航迹real_trajectory以及第一个点的信息first_point1
def readBegin():
    db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # 查询起点
    try:
        cursor.execute("SELECT ID,MMSI,BaseDateTime,LAT,LON,SOG,COG FROM ais_2017 WHERE ID= '{0}';".format(ID))
        firstPoint = cursor.fetchall()[0]
        global MMSISelect
        MMSISelect = firstPoint[1]
        global BaseDateTimeSelect
        BaseDateTimeSelect = firstPoint[2]
    except:
        print("读取起点错误")
    # 查询真实航迹
    try:
        # 执行SQL语句,获得预测真实航迹
        cursor.execute(
            "SELECT ID,MMSI,BaseDateTime,LAT,LON,SOG,COG FROM ais_2017 WHERE  MMSI = '{0}' order by BaseDateTime ;".format(
                MMSISelect))
        # 执行SQL语句,获得起点信息
        realTrajectory = list(cursor.fetchall())
    except:
        print
        "Error: unable to fecth data"
    firstPoint = list(firstPoint)
    if (firstPoint[6] < -180):
        firstPoint[6] = 360 + firstPoint[6]
    if (firstPoint[6] > 180):
        firstPoint[6] = firstPoint[6] - 360
    print("===============初始点和真实轨迹成功查询====================")
    print(firstPoint)
    return realTrajectory, firstPoint


# 根据条件获取周围相似AIS点
def getS0(firstPoint):
    # 初始点的维度、经度、速度、航向
    global MMSISelect
    first_LAT = firstPoint[3]
    first_LON = firstPoint[4]
    first_SOG = firstPoint[5]
    first_COG = firstPoint[6]
    db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # SQL 查询语句,查询起始点附近0.05°的船只信息
    sql_result = "SELECT ID,MMSI,BaseDateTime,LAT,LON,SOG,COG FROM ais_2017 WHERE lat BETWEEN {0} AND {1} AND lon BETWEEN {2} AND {3} AND MMSI!='{4}';".format(
        first_LAT - 0.03, first_LAT + 0.03, first_LON - 0.03, first_LON + 0.03, MMSISelect)
    try:
        # 执行SQL语句,获得预测真实航迹
        cursor.execute(sql_result)
        result_trajectory = list(cursor.fetchall())
    except:
        print
        "查询周围AIS点错误"
    resultTrajectory = []
    for i in range(len(result_trajectory)):
        result_trajectory[i] = list(result_trajectory[i])
        if (result_trajectory[i][6] < -180):
            result_trajectory[i][6] = 360 + result_trajectory[i][6]
        if (result_trajectory[i][6] > 180):
            result_trajectory[i][6] = result_trajectory[i][6] - 360
        if abs(first_COG - result_trajectory[i][6]) < 30 or abs(first_COG - result_trajectory[i][6]) > 330:
            if abs(first_SOG - result_trajectory[i][5]) < 5:  # 航速差小于5节
                resultTrajectory.append(result_trajectory[i][1:])
    print("=================周围找初始类完成===========================")
    S0 = []
    for i in range(len(resultTrajectory)):
        if getPointDistance([first_LAT, first_LON], resultTrajectory[i][2:4]) < 1:
            S0.append(resultTrajectory[i])
    temp = {}
    for MMSI, BaseDateTime, LAT, LON, SOG, COG in S0:
        if MMSI not in temp:  # we see this key for the first time
            temp[MMSI] = (MMSI, BaseDateTime, LAT, LON, SOG, COG)
        else:
            # 找出相同MMSI中距离预测点最近的那个点
            if getPointDistance([first_LAT, first_LON], [LAT, LON]) < getPointDistance([first_LAT, first_LON],
                                                                                       temp[MMSI][2:4]):
                temp[MMSI] = (MMSI, BaseDateTime, LAT, LON, SOG, COG)
    S0 = list(temp.values())
    print("============得到初始类S0=============")
    return S0


# 提取S0以及firstPoint前置、后置轨迹,时间间隔为30S，前后各30min
def getTrajectory(S0, firstPoint):
    del firstPoint[0]  # 将起始点的数据ID删除
    backTrajectory = []
    forwardTrajectory = []
    print("SO中一共会有", len(S0), "条航迹")
    S0.append(firstPoint)
    S1 = [] #用于存储周围轨迹的起点
    for i in range(len(S0)):
        now_time = S0[i][1]
        begin_time_date = now_time - datetime.timedelta(hours=2)
        begin_time = datetime.datetime.strftime(begin_time_date, '%Y-%m-%d %H:%M:%S')
        end_time_date = now_time + datetime.timedelta(hours=2)
        end_time = datetime.datetime.strftime(end_time_date, '%Y-%m-%d %H:%M:%S')
        db = MySQLdb.connect("localhost", "root", "123456", "ais", charset='utf8')
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # SQL 查询语句
        sql_S0 = "SELECT ID,MMSI,BaseDateTime,LAT,LON,SOG,COG FROM ais_2017 WHERE  MMSI = '{0}' and  lat BETWEEN {1} AND {2} AND lon BETWEEN {3} AND {4} and BaseDateTime between '{5}' and '{6}' order by BaseDateTime;".format(
            S0[i][0], S0[i][2] - 2, S0[i][2] + 2, S0[i][3] - 2, S0[i][3] + 2, begin_time, end_time)
        try:
            # 执行SQL语句,获得周围目标船舶的航迹
            cursor.execute(sql_S0)
            S0_tajectory = cursor.fetchall()
        except:
            print("从数据库中轨迹提取出现错误")
        first_time = S0[i][1]
        date = [l[2] for l in S0_tajectory]
        # 将时间转换为时间戳
        dateStamp = []
        for k in range(len(date)):
            dateStamp.append(int(date[k].timestamp()))
        LAT = [k[3] for k in S0_tajectory]
        LON = [k[4] for k in S0_tajectory]
        SOG = [k[5] for k in S0_tajectory]
        COG = [k[6] for k in S0_tajectory]
        for k in range(len(COG)):
            if (COG[k] < -180):
                COG[k] = 360 + COG[k]
            if (COG[k] > 180):
                COG[k] = COG[k] - 360
        X = pd.date_range(start=first_time - datetime.timedelta(minutes=30), periods=121, freq='30S')
        X = list(X)
        XStamp = []
        for h in range(len(X)):
            XStamp.append(int(X[h].to_pydatetime().timestamp()))
        if (dateStamp[len(dateStamp) - 1] - dateStamp[0]) < 3000:
            print("航迹持续时间小于50min，舍弃轨迹", i)
            continue
        if (max(np.diff(dateStamp))>600):
            print("轨迹存在相邻时间间隔大于10min，舍弃轨迹",i)
            continue
        if (XStamp[0] < dateStamp[0])-600 or (dateStamp[len(dateStamp) - 1] < XStamp[120]-600):
            print('轨迹', i, '左右边界不足，需要外插')
        S1.append(S0[i])
        f_LAT = interpolate.interp1d(dateStamp, LAT, kind="linear", fill_value='extrapolate')
        f_LON = interpolate.interp1d(dateStamp, LON, kind="linear", fill_value='extrapolate')
        f_SOG = interpolate.interp1d(dateStamp, SOG, kind="linear", fill_value='extrapolate')
        f_COG = interpolate.interp1d(dateStamp, COG, kind="linear", fill_value='extrapolate')
        LATS = f_LAT(XStamp)
        LONS = f_LON(XStamp)
        SOGS = f_SOG(XStamp)
        COGS = f_COG(XStamp)
        if min(SOGS)<1:
            print("轨迹",i,"插值之后航速过小，该船可能将要处于停泊状态，删除该轨迹")
            continue
        back = []
        forward = []
        for j in range(0, 60):
            back.append([S0[i][0], X[j], LATS[j], LONS[j], SOGS[j], COGS[j]])
        backTrajectory.append(back)
        for m in range(61, 121):
            forward.append([S0[i][0], X[m], LATS[m], LONS[m], SOGS[m], COGS[m]])
        forwardTrajectory.append(forward)
        print("轨迹", i, "完成！")
    backSelect = backTrajectory[len(backTrajectory) - 1]
    forwardSelect = forwardTrajectory[len(backTrajectory) - 1]
    del backTrajectory[len(backTrajectory) - 1]
    del forwardTrajectory[len(forwardTrajectory) - 1]
    # del S1[-1:]
    print("================轨迹提取完成=================")
    return backTrajectory, forwardTrajectory, backSelect, forwardSelect, S1


# 依据backTrajectory+forwardTrajectory对轨迹进行聚类,60个2维AIS点数据+1个label，
def clusterKmeans(backTrajectory, forwardTrajectory, backSelect, forwardSelect):
    trajectory = []
    trajectoryChain = []  # 得到60*4=240维的数据
    MMSI = []  # 记录每条航迹的MMSI
    SOGANDCOG = []
    for i in range(len(backTrajectory)):
        MMSI.append(backTrajectory[i][0][0])
        trajectory.append(backTrajectory[i] + forwardTrajectory[i])
    for i in range(len(trajectory)):
        trajectorySingle = []
        SOGANDCOGSingle = []
        for j in range(len(trajectory[i])):
            del trajectory[i][j][0:2]
            SOGANDCOGSingle += trajectory[i][j][2:]
            del trajectory[i][j][2:]
            trajectorySingle += trajectory[i][j]
        trajectoryChain.append(trajectorySingle)
        SOGANDCOG.append(SOGANDCOGSingle)
    # for i in range(0,len(trajectory)):
    #     for j in range(0,len(trajectory[i])):
    #         trajectorySingle.extend(trajectory[i][j][2:])
    #     trajectoryChain.append(trajectorySingle)
    # 数据标准化
    # trajectoryChainStand = StandardScaler().fit_transform(trajectoryChain)
    # pca = PCA(n_components= 24)
    # trajectoryChainPCA = pca.fit_transform(trajectoryChain)
    eps = 5
    COGS = []
    for i in SOGANDCOG:
        COGS.append(i[1::2])
    COGSStand = StandardScaler().fit_transform(COGS)
    bgin1 = time.time()
    length = len(trajectoryChain)
    threads = []
    result = []
    try:
        for i in range(int(length/100)):
            threads.append(MyThread(clusterDistance, args=("t"+str(i),COGS,trajectoryChain,i*100,(i+1)*100)))
        threads.append(MyThread(clusterDistance, args=("t"+str(i+1),COGS,trajectoryChain,(i+1)*100,length)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for thread in threads:
            result+=thread.get_result()
    except:
        print("线程异常")
    result = np.array(result)
    result = np.triu(result)
    result += result.T-np.diag(result.diagonal())
    end1 = time.time()
    print("计算相似性矩阵用时：",end1-bgin1)
    begin = time.time()
    # 构建空列表，用于保存不同参数组合下的结果
    res = []
    for eps in np.arange(1,100,1):
        # 迭代不同的min_samples值
        for min_samples in range(2, 10):
            dbscan = DBSCAN(min_samples=min_samples, eps=eps, leaf_size=1000, metric='precomputed')
            label = dbscan.fit(np.array(result))
            # 统计各参数组合下的聚类个数（-1表示异常点）
            n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
            # 异常点的个数
            outliners = np.sum(np.where(dbscan.labels_ == -1, 1, 0))
            # 统计每个簇的样本个数
            stats = str(pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values)
            res.append({'eps': eps, 'min_samples': min_samples, 'n_clusters': n_clusters, 'outliners': outliners,
                        'stats': stats})
        print("eps：",eps,"结束")
    df = pd.DataFrame(res)
    label = DBSCAN(min_samples=6, eps=72, leaf_size=1000, metric='precomputed').fit_predict(np.array(result))
    end = time.time()
    print("聚类用时：",end-begin)
    print("聚类类别：",label)
    # label = list(GaussianMixture(n_components=5,).fit_predict(trajectoryChain))
    # data = []
    # for back, foward in zip(backTrajectory, forwardTrajectory):
    #     data.append(back + foward)
    # label = KMeans(n_clusters=5).fit_predict(trajectoryChain)
    # kclusterer = KMeansClusterer(5, repeats=25)
    # label = kclusterer.cluster(np.array(trajectoryChain), assign_clusters=True)
    print("GMM聚类任务完成")
    # ==========================给轨迹打上label=====================
    trajectoryLabel = []
    backTrajectoryLabel = []
    forwardTrajectoryLabel = []
    for i in range(0, len(MMSI)):
        trajectoryLabel.append(list(trajectoryChain[i] + [[label[i]]]))
        backTrajectoryLabel.append(list(backTrajectory[i] + [[label[i]]]))
        forwardTrajectoryLabel.append(list(forwardTrajectory[i] + [[label[i]]]))
    # trajectoryLabel = np.array(trajectoryLabel)
    # ===========================展示聚类结果======================
    # for i in range(0,240,2):
    #     plt.scatter(trajectoryLabel[:,i+1],trajectoryLabel[:,i],c=sum(trajectoryLabel[:,240],[]),cmap=plt.cm.Spectral)
    for single in trajectoryLabel:
        # 不展示异常轨迹
        # if(single[len(single) - 1][0]==-1):
        #     continue
        c = num_to_color(single[len(single) - 1][0])
        plt.plot([single[2 * i + 1] for i in range(0, 120, 2)], [single[2 * i] for i in range(0, 120, 2)], color=c)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    plt.plot([x[3] for x in backSelect], [x[2] for x in backSelect], color='black')
    plt.plot([x[3] for x in forwardSelect], [x[2] for x in forwardSelect], color='black')
    plt.title('cluster,pes='+str(eps))
    plt.savefig(dirs + '/聚类'+str(eps)+'.png', dpi=1080)
    plt.show()
    return backTrajectoryLabel, forwardTrajectoryLabel, SOGANDCOG


def clusterDistance(name,cogs,trajectoryChain,start,end):
    distanc = []
    distanc1 = []
    a = 0.5
    for i in range(start,end,1):
        t1 = time.time()
        for x in range(i+1):
            distanc1.append(0)
        for j in range(i + 1, len(trajectoryChain)):
            sum = 0
            for k in range(0, 240, 10):
                # sum+= math.sqrt((trajectoryChain[i][k]-trajectoryChain[j][k])**2+(trajectoryChain[i][k+1]-trajectoryChain[j][k+1])**2)
                disSOG = abs(angle(trajectoryChain[i][k:k + 2], trajectoryChain[j][k:k + 2]))*(10**5)
                # sum+=abs(cogs[i][k]-cogs[j][k])/360*getPointDistance(trajectoryChain[i][2*k:2*k+2],trajectoryChain[j][2*k:2*k+2])
                # d1 = math.sqrt((trajectoryChain[i][k]-trajectoryChain[j][k])**2+(trajectoryChain[i][k+1]-trajectoryChain[j][k+1])**2)
                # d2 = math.sqrt((trajectoryChain[i][k+8]-trajectoryChain[j][k+8])**2+(trajectoryChain[i][k+9]-trajectoryChain[j][k+9])**2)
                # b1 = math.sqrt((trajectoryChain[i][k+8]-trajectoryChain[i][k])**2+(trajectoryChain[i][k+9]-trajectoryChain[i][k+1])**2)
                # b2 = math.sqrt((trajectoryChain[j][k+8]-trajectoryChain[j][k])**2+(trajectoryChain[j][k+9]-trajectoryChain[j][k+1])**2)
                # Angle = abs((cogs[i][k]+cogs[i][k+8])/2-(cogs[j][k]+cogs[j][k+8])/2)
                # if Angle>math.pi/2:
                #     Angle = math.pi/2
                # sum+=abs(cogs[i][k]-cogs[j][k])*math.sqrt((trajectoryChain[i][k]-trajectoryChain[j][k])**2+(trajectoryChain[i][k+1]-trajectoryChain[j][k+1])**2)
                # sum+=abs(cogs[i][k]-cogs[j][k])*a+(1-a)*math.sqrt((trajectoryChain[i][k]-trajectoryChain[j][k])**2+(trajectoryChain[i][k+1]-trajectoryChain[j][k+1])**2)
                # sum+= d1+d2+(b1*math.sin(Angle)+b2*math.sin(Angle))/2
                # sum+=abs(cogs[i][k]-cogs[j][k])
                dis = abs(trajectoryChain[i][k]-trajectoryChain[j][k])+abs(trajectoryChain[i][k+1]-trajectoryChain[j][k+1])*(10**2)
                sum+=disSOG*0.6 + dis*0.4
            distanc1.append(sum)
        distanc.append(distanc1)
        distanc1=[]
        t2 = time.time()
        print(name,"线程,",i, "号轨迹用时：", t2 - t1)
    return distanc


# 采用KNN算法分类，返回selectTrajectory的label
def classifyKNN(backTrajectoryLabel1, backSelect):
    backTrajectoryLabel = copy.deepcopy(backTrajectoryLabel1)
    backCopy = []
    selectCopy = []
    label = []
    for single in backTrajectoryLabel:
        if(single[60][0]==-1):
            continue
        label.append(single[60][0])
        del single[60]
        backCopy.append(sum(single, []))
    for single in backSelect:
        selectCopy += single[2:4]
    k = KNNvalidation(dataSet=backCopy,label=label)
    begin = time.time()
    print("开始分类时间：", begin)
    # clf = KNeighborsClassifier(n_neighbors=k,metric=getPointDistanceTrajectoryBack,n_jobs=-1)
    # clf.fit(backCopy, label)
    # kind = clf.predict([selectCopy])
    kind = classify(selectCopy, backCopy, label, k)
    end = time.time()
    print("结束分类时间：", end, "共用时：", end - begin)
    print("=================KNN分类完成=================")
    return kind

# 交叉验证KNN分类时K值的大小
def KNNvalidation(dataSet,label):
    parameters = {'n_neighbors':range(1,21)}
    knn = KNeighborsClassifier()  # 注意：这里不用指定参数
    # 通过GridSearchCV来搜索最好的K值。这个模块的内部其实就是对每一个K值进行评估
    clf = GridSearchCV(knn, parameters, cv=5)  # 5折
    clf.fit(dataSet, label)
    # 输出最好的参数以及对应的准确率
    print("最终最佳准确率：%.2f" % clf.best_score_, "最终的最佳K值", clf.best_params_)
    return clf.best_params_['n_neighbors']

# 计算backTrajectory与backSelect的距离，根据权重和forwardTrajectory预测目标船舶将来的航迹
def getPredict(backTrajectoryLabel1, forwardTrajectoryLabel1, backSelect1, label, forwardSelect1, firstPoint,
               SOGANDCOG, SOGFirst,S1):
    # backTrajectoryLabel = copy.deepcopy(backTrajectoryLabel1)
    # forwardTrajectory = copy.deepcopy(forwardTrajectory1)
    # backSelect = copy.deepcopy(backSelect1)
    backTrajectoryLabel = []
    forwardTrajectoryLabel = []
    SOGs = []  # 相同label船舶的SOG
    S1Label = [] # 用于存储与目标船舶同类型的周围船舶的起点
    for i in range(len(backTrajectoryLabel1)):
        if backTrajectoryLabel1[i][60] == label:  # 获取与目标船舶相同label的轨迹
            backTrajectoryLabel.append(sum(backTrajectoryLabel1[i], []))
            forwardTrajectoryLabel.append(sum(forwardTrajectoryLabel1[i], []))
            SOGs.append(SOGANDCOG[i][::2])
            S1Label.append(S1[i])

    # 画出与目标船舶同类的轨迹分布图
    for i in range(len(backTrajectoryLabel)):
        plt.plot([backTrajectoryLabel[i][2 * x + 1] for x in range(60)],
                 [backTrajectoryLabel[i][2 * x] for x in range(60)], color='red' )  # 后置轨迹红色
        plt.plot([forwardTrajectoryLabel[i][2 * x + 1] for x in range(60)],
                 [forwardTrajectoryLabel[i][2 * x] for x in range(60)], color='green')  # 前置轨迹绿色
    plt.plot([single[3] for single in backSelect1], [single[2] for single in backSelect1], color='black')
    plt.plot([single[3] for single in forwardSelect1], [single[2] for single in forwardSelect1], color='black')
    plt.plot(firstPoint[3], firstPoint[2], color='black', marker='x')  # 预测的起点
    plt.title("same label")
    plt.show()

    backSelect = sum([i[2:4] for i in backSelect1], [])  # 获取目标船舶的轨迹
    weigth = []
    # for single in backTrajectoryLabel:
    #     distance = getPointDistanceTrajectoryFoward(single[:120], backSelect)
    #     weigth.append(1 / distance)
    # for i in weigth:
    #     sum_weight = sum_weight + i
    # weightBack = [x/sum_weight for x in weigth]
    print("与select船舶相同类别轨迹的数目是：", len(backTrajectoryLabel))
    preTrajectory = []
    # ======================首轮预测=======================
    trajectoryDistance = []
    singlePredict = []
    moves = []
    for single, singleFoward,oneSOG,S1LabelSingle in zip(backTrajectoryLabel, forwardTrajectoryLabel,SOGs,S1Label):
        distance, move = getDistancePrediction(single[100:120], backSelect[100:120], singleFoward[0:20],oneSOG[50:60], SOGFirst,
                                               firstPoint[2:4],S1LabelSingle[2:4]) #此处应该设置周围船舶的起点位置
        trajectoryDistance.append(distance)
        moves.append(move)
    min_index, min_num = find_min_nums(trajectoryDistance, 10)
    sumWeight = 0.0
    for i in range(len(forwardTrajectoryLabel)):
        if i in min_index:
            singlePredict.append(moves[i])
            sumWeight += trajectoryDistance[i]
    preTrajectory += addList(singlePredict)
    # =====================使用首轮预测的5min作为下一轮DTW算法的比较量，依次迭代继续预测剩下的25min================
    for i in range(0, len(forwardTrajectoryLabel[0]) - 21, 20):
        trajectoryDistance = []
        singlePredict = []
        moves = []
        for single, oneSOG in zip(forwardTrajectoryLabel,SOGs):
            distance, move = getDistancePrediction(single[i:i + 20], preTrajectory[-20:],
                                                   single[i + 20:i + 40],
                                                   oneSOG[(60 + int(i / 2)):(70 + int(i / 2))],
                                                   SOGFirst,
                                                   preTrajectory[-2:], single[i + 18:i+20])
            trajectoryDistance.append(distance)
            moves.append(move)
        min_index, min_num = find_min_nums(trajectoryDistance, 10)
        sumWeight = 0.0
        for j in range(len(forwardTrajectoryLabel)):
            if j in min_index:
                singlePredict.append(moves[j])
                sumWeight += trajectoryDistance[j]
        preTrajectory += addList(singlePredict)
    return preTrajectory


# 计算误差，包括绝对误差和相对误差，画出对应轨迹图
def getError(preTrajectory, forwardSelect, firstPoint):
    absoluteError = []
    relativeError = []
    route = []
    for i in range(len(forwardSelect)):
        absoluteError += [getPointDistance(forwardSelect[i][2:4], preTrajectory[2 * i:2 * i + 2])]
    route.append(getPointDistance(forwardSelect[0][2:4], firstPoint[2:4]))
    for i in range(1, len(forwardSelect)):
        route.append(getPointDistance(forwardSelect[i][2:4], forwardSelect[i - 1][2:4]) + route[i - 1])
        relativeError.append([absoluteError[i - 1] / route[i - 1]])
    relativeError.append([absoluteError[len(absoluteError) - 1] / route[len(route) - 1]])
    # =============================画图=================================
    plt.plot([x[3] for x in forwardSelect], [x[2] for x in forwardSelect], color='green')  # 真实点用绿色
    plt.plot([x[3] for x in forwardSelect], [x[2] for x in forwardSelect], 'o', color='green',label="real")  # 真实点用绿色
    plt.plot([preTrajectory[2 * i + 1] for i in range(0, 60, 1)], [preTrajectory[2 * i] for i in range(0, 60, 1)],
             color='red')  # 预测点用红色
    plt.plot([preTrajectory[2 * i + 1] for i in range(0, 60, 1)], [preTrajectory[2 * i] for i in range(0, 60, 1)], 'o',
             color='red',label="predict")  # 预测点用红色
    plt.plot(firstPoint[3], firstPoint[2], color='black', marker='x')  # 预测的起点
    plt.title('Predicted results')
    plt.savefig(dirs + '/预测结果.png', dpi=1080)
    plt.legend()
    plt.show()
    plt.plot(absoluteError)
    plt.title('absoluteError/nm')
    plt.savefig(dirs + '/绝对误差.png', dpi=1080)
    plt.show()
    plt.plot(relativeError)
    plt.title('relativeError')
    plt.savefig(dirs + '/相对误差', dpi=1080)
    plt.show()
    return absoluteError, relativeError

# LSTM预测。用来做实验对比，传入历史数据，目标船舶轨迹
def lstmPrediction(history, selectBack):
    prediction = [] # 用来存放LSTM预测得到的轨迹
    return prediction

# ==================工具函数===============
# 计算A、B两点实际距离
def getPointDistance(pointA=[], pointB=[]):
    distance = geodesic((pointA[0], pointA[1]), (pointB[0], pointB[1])).nm
    return distance


# 聚类时计算a,b两条轨迹之间的DTW距离
def d(x,y):
    return np.sum((x-y)**2)

# 二维数据的dtw距离（时间复杂度很高）
def dtw_distance_two(ts_a,ts_b):
    mww = 10000
    M, N = np.shape(ts_a)[1], np.shape(ts_b)[1]
    cost = np.ones((M, N))
    # Initialize the first row and column
    cost[0, 0] = d(ts_a[:,0], ts_b[:,0])
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + d(ts_a[:,i], ts_b[:,0])
    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + d(ts_a[:,0], ts_b[:,j])
    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - mww), min(N, i + mww)):
            choices = cost[i-1, j-1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(ts_a[:,i], ts_b[:,j])
    # Return DTW distance given window
    return cost[-1, -1]

# 一维数据的dtw距离（传入降维之后的一维数据）
def dtw_distance_one(ts_a, ts_b, d=lambda x, y: abs(x - y), mww=10000):
    """Computes dtw distance between two time series

    Args:
        ts_a: time series a
        ts_b: time series b
        d: distance function
        mww: max warping window, int, optional (default = infinity)

    Returns:
        dtw distance
    """

    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - mww), min(N, i + mww)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window
    return cost[-1, -1]

###通过KNN进行分类,输入目标船舶、周围船舶、船舶标签、参数K
def classify(input, dataSet, label, k):
    ####计算欧式距离
    dist = []
    for single in dataSet:
        dist.append(getDistanceClassfiy(single,input))
    ##对距离进行排序
    sortedDistIndex = np.argsort(dist)  ##argsort()根据元素的值从小到大对元素进行排序，返回下标
    classCount = {}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        ###对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1  # 字典中的键voteLabel,若没有则添加键的默认值位0
    ###选取出现的类别次数最多的类别
    maxCount = 0
    for key, value in classCount.items():  # .item遍历字典中所有键
        if value > maxCount:
            maxCount = value
            classes = key  # 返回出现次数最多的类别
    return classes


# # 计算两条轨迹之间的距离,采用时间间隔为5×30S=150S，在预测阶段使用
# def getPointDistanceTrajectoryFoward(trajectoryA=[], trajectoryB=[]):
#     distance = 0.0
#     for i in range(0, len(trajectoryA), 10):
#         distance += getPointDistance(trajectoryA[i:i + 2], trajectoryB[i:i + 2])
#     return distance


# 输入20维即5min的数据，预测阶段时计算轨迹之间距离，以及航迹位移,
# A是周围轨迹,B是目标船舶，A和B是用来做相似度比较的，OneSog为周围轨迹的航速，SogFirst为起始点的速度，start为起始点的位置,C为相似的轨迹的将来变化趋势
def getDistancePrediction(A, B, C, OneSog, SogFirst, start, pastPoint):
    distance = 0
    move = []
    sumSOG = 0
    for i in range(0, len(A)-3, 2):
        distance += abs(angle([A[i + 2]-A[i],A[i+3]-A[i+1]], [B[i + 2]-B[i],B[i+3]-B[i+1]]))
    for i in range(len(OneSog)):
        sumSOG += OneSog[i]
    avgSOG = sumSOG / len(OneSog)
    move.append([(C[0]-pastPoint[0])*SogFirst/avgSOG+start[0], (C[1]-pastPoint[1])*SogFirst/avgSOG+start[1]])
    for i in range(0, 18, 2):
        move.append([(C[i + 2] - C[i]) * SogFirst / avgSOG + move[int(i / 2)][0],
                     (C[i + 3] - C[i + 1]) * SogFirst / avgSOG + move[int(i / 2)][1]])
    return distance, move

# 在分类时确定A、B两条轨迹之间的距离
def getDistanceClassfiy(A, B):
    distance = 0
    for i in range(0, len(A)-3, 2):
        distance += abs(angle([A[i + 2]-A[i],A[i+3]-A[i+1]], [B[i + 2]-B[i],B[i+3]-B[i+1]]))
    return distance

# 分类时，计算轨迹之间的欧氏距离,输入120维数据，A是周围轨迹，B是目标船轨迹，采样间隔为5min
def getDistanceBetweenTrajectory(A, B):
    distance = 0
    vectorA = []
    vectorB = []
    for i in range(0, len(A), 20):
        vectorA.append([A[i + 18] - A[i], A[i + 19] - A[i + 1]])
        vectorB.append([B[i + 18] - B[i], B[i + 19] - B[i + 1]])
    for i in range(len(vectorA)):
        distance += abs(angle(vectorA[i], vectorB[i]))
    return distance


# 从DTW的list当中找到值最小的是个元素的索引和值
def find_min_nums(nums, find_nums):
    if len(nums) == len(list(set(nums))):
        # 使用heapq
        min_number = heapq.nsmallest(find_nums, nums)
        min_num_index = list(map(nums.index, min_number))
    else:
        # 使用deepcopy
        nums_copy = copy.deepcopy(nums)
        max_num = max(nums) + 1
        min_num_index = []
        min_number = []
        for i in range(find_nums):
            num_min = min(nums_copy)
            num_index = nums_copy.index(num_min)
            min_number.append(num_min)
            min_num_index.append(num_index)
            nums_copy[num_index] = max_num
    return min_num_index, min_number


# 二维list逐行相加
def addList(A):
    B = []
    C = []
    for j in A:
        B.append(np.array(j).reshape(1, 20)[0])
    for i in range(20):
        tem = 0
        for j in range(len(B)):
            tem += B[j][i]
        C.append(tem / 10)
    return C


# 根据聚类后不同label，返回不同颜色的字符串
def num_to_color(num):
    numbers = {
        0: "dodgerblue",
        1: "cyan",
        2: "brown",
        3: "gray",
        4: "yellow",
        -1:"pink",
    }
    return numbers.get(num, None)


# 两个list对应元素求和
def list_add(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i] + b[i])
    return c


# 根据两个向量，求出它们之间的夹角
def angle(vec1, vec2, deg=False):
    _angle = np.arctan2(np.abs(np.cross(vec1, vec2)), np.dot(vec1, vec2))
    if deg:
        _angle = np.rad2deg(_angle)
    return _angle

# 测试dbsacn距离函数
def DBSCANDistance(A,B):
    sum = 0
    for a,b in zip(np.array(A).reshape(120,2),np.array(B).reshape(120,2)):
        sum+=(a[0]-b[0])**2
    return sum

def out():
    realTrajectory, firstPoint = readBegin()
    SOGFirst = firstPoint[5]
    S0 = getS0(firstPoint)
    backTrajectory, forwardTrajectory, backSelect, forwardSelect, S1 = getTrajectory(S0, firstPoint)
    backTrajectoryLabel, forwardTrajectoryLabel, SOGANDCOGALL = clusterKmeans(backTrajectory, forwardTrajectory,
                                                                              backSelect,
                                                                              forwardSelect)
    label = classifyKNN(backTrajectoryLabel, forwardSelect)
    print("目标船舶属于第",label,"类")
    predictTrajectory = getPredict(backTrajectoryLabel, forwardTrajectoryLabel, backSelect, label, forwardSelect,
                                   firstPoint, SOGANDCOGALL, SOGFirst,S1)
    absoluteError, relativeError = getError(predictTrajectory, forwardSelect, firstPoint)


out()
