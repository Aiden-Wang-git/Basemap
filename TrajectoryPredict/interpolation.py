from pylab import mpl, datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import matplotlib.ticker as mtick
# 进行三次样条拟合
# 传入轨迹数据AIS的list，需要插值的时间戳
from TrajectoryCluster.aisPoint import AIS


def interpolation3(trajectory, inter_time):
    MMSI = trajectory[0].MMSI
    Length = trajectory[0].Length
    Width = trajectory[0].Width
    VesselType = trajectory[0].VesselType
    # 先统计原先轨迹时间戳，转换为从0开始的list
    time = []
    SOGs = []
    LATs = []
    LONs = []
    COGs = []
    start_time = trajectory[0].BaseDateTime
    for point in trajectory:
        time.append((point.BaseDateTime - start_time).total_seconds())
        SOGs.append(point.SOG)
        COGs.append(point.COG)
        LATs.append(point.LAT)
        LONs.append(point.LON)
    # =========插值SOG===========
    ipo3 = spi.splrep(time, SOGs, k=3)  # 样本点导入，生成参数
    SOG_3 = spi.splev(inter_time, ipo3)  # 根据观测点和样条参数，生成插值
    # =========插值SOG===========
    ipo3 = spi.splrep(time, COGs, k=3)  # 样本点导入，生成参数
    COG_3 = spi.splev(inter_time, ipo3)  # 根据观测点和样条参数，生成插值
    # =========插值SOG===========
    ipo3 = spi.splrep(time, LATs, k=3)  # 样本点导入，生成参数
    LAT_3 = spi.splev(inter_time, ipo3)  # 根据观测点和样条参数，生成插值
    # =========插值SOG===========
    ipo3 = spi.splrep(time, LONs, k=3)  # 样本点导入，生成参数
    LON_3 = spi.splev(inter_time, ipo3)  # 根据观测点和样条参数，生成插值
    for i in range(len(SOG_3)):
        trajectory.append(
            AIS(ID='',
                MMSI=MMSI,
                BaseDateTime=start_time + datetime.timedelta(seconds=inter_time[i]),
                LAT=LAT_3[i],
                LON=LON_3[i],
                SOG=SOG_3[i],
                COG=COG_3[i],
                Length=Length,
                Width=Width,
                VesselType=VesselType))
    trajectory.sort(key=lambda point: point.BaseDateTime)
    # draw_interpolation(SOG_3, COG_3, LAT_3, LON_3, SOGs, LATs, LONs, COGs, inter_time, time)
    return trajectory


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def draw_interpolation(SOG_3, COG_3, LAT_3, LON_3, SOGs, LATs, LONs, COGs, inter_time, time):
    # plt.ticklabel_format(axis="y", style='plain', scilimits=(0, 0))
    plt.plot(time, LATs, 'o-')
    plt.plot(time, LATs, 'go', label='样本点')
    plt.plot(inter_time, LAT_3, 'ro', label='插值点')
    # plt.set_ylim(LATs.min() - 1, LATs.max() + 1)
    plt.ylabel('指数')
    plt.title('三次样条插值-LAT')
    plt.legend()
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time, LONs, 'o-')
    ax.plot(time, LONs, 'go', label='样本点')
    ax.plot(inter_time, LON_3, 'ro', label='插值点')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    # plt.set_ylim(LONs.min() - 1, LONs.max() + 1)
    plt.ylabel('指数')
    plt.title('三次样条插值-LON')
    plt.legend()
    plt.show()
#
#
#
#
#
# # 数据准备
# X = np.arange(-np.pi, np.pi, 1)  # 定义样本点X，从-pi到pi每次间隔1
# Y = np.sin(X)  # 定义样本点Y，形成sin函数
# new_x = np.arange(-np.pi, np.pi, 0.1)  # 定义差值点
#
# # 进行样条差值
# import scipy.interpolate as spi
#
# # 进行一阶样条插值
# ipo1 = spi.splrep(X, Y, k=1)  # 样本点导入，生成参数
# iy1 = spi.splev(new_x, ipo1)  # 根据观测点和样条参数，生成插值
#
# # 进行三次样条拟合
# ipo3 = spi.splrep(X, Y, k=3)  # 样本点导入，生成参数
# iy3 = spi.splev(new_x, ipo3)  # 根据观测点和样条参数，生成插值
#
# ##作图
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
#
# ax1.plot(X, Y, 'o', label='样本点')
# ax1.plot(new_x, iy1, label='插值点')
# ax1.set_ylim(Y.min() - 1, Y.max() + 1)
# ax1.set_ylabel('指数')
# ax1.set_title('线性插值')
# ax1.legend()
#
#
# ax2.plot(X, Y, 'o', label='样本点')
# ax2.plot(new_x, iy3, label='插值点')
# ax2.set_ylim(Y.min() - 1, Y.max() + 1)
# ax2.set_ylabel('指数')
# ax2.set_title('三次样条插值')
# ax2.legend()
#
# plt.show()
