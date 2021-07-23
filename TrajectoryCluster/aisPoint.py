from __future__ import unicode_literals, absolute_import
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Float

Base = declarative_base()


class AIS(Base):
    __tablename__ = "ais_2017"

    ID = Column(Integer, primary_key=True)
    MMSI = Column(String(64))
    BaseDateTime = Column(DateTime)
    LAT = Column(Float(10))
    LON = Column(Float(10))
    SOG = Column(Float(10))
    COG = Column(Float(10))
    Length = Column(Float(10))
    Width = Column(Float(10))
    VesselName = Column(String(64),default="")
    VesselType = Column(Integer,default=-1)
    Draft = Column(Float(10),default=0)
    Cargo = Column(Float(10),default=0)

    def __init__(self, ID, MMSI, BaseDateTime, LAT, LON, SOG, COG, Length, Width,VesselType):
        self.ID = ID
        self.MMSI = MMSI
        self.BaseDateTime = BaseDateTime
        self.LAT = LAT
        self.LON = LON
        self.SOG = SOG
        self.COG = COG
        self.Length = Length
        self.Width = Width
        self.VesselType = VesselType

