from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from TrajectoryCluster.aisPoint import AIS
from sqlalchemy import and_
from sqlalchemy import func
from TrajectoryCluster.trajectory import Trajectory
import datetime

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

# 读取研究范围内所有航速大于1的AIS点
datas = session.query(AIS).filter(
    and_(AIS.LAT >= 33.5, AIS.LAT <= 33.6, AIS.LON >= -118.35, AIS.LON <= -118.25, AIS.SOG >= 1)).order_by(AIS.MMSI,
                                                                                                           AIS.BaseDateTime).all()
# 将航迹依据MMSI分开
# 将间隔时间大于10min的轨迹断开
MMSI = datas[0].MMSI
trajectories = []
trajectory = Trajectory(MMSI)
for data in datas:
    if data.MMSI == MMSI:
        if trajectory.getLength() == 0 or (data.BaseDateTime - trajectory.points[trajectory.getLength() - 1].BaseDateTime).seconds <= 600:
            trajectory.add_point(data)
            continue
    trajectories.append(trajectory)
    MMSI = data.MMSI
    trajectory = Trajectory(MMSI)
    trajectory.add_point(data)

print(len(datas))

session.close()
