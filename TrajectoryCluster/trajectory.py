class Trajectory:
    points = []
    count = 0
    MMSI = "null"

    def add_point(self, point):
        self.points.append(point)
        self.count = self.count + 1

    def setMMSI(self, MMSI):
        self.MMSI = MMSI

    def getLength(self):
        return len(self.points)

    def __init__(self, MMSI):
        self.MMSI = MMSI
        self.points = []
