import numpy as np
import csv

dict = np.load("trajectories_process2.npy", allow_pickle=True).item()
trajectories = []
for key in dict:
    trajectories.append(dict[key])

labels = list(np.load('labels.npy', allow_pickle=True))

for i in range(len(trajectories)):
    label = labels[i]
    trajectory = trajectories[i]
    with open(f"label{label}.csv", 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        point = trajectory.points[0]
        writer.writerow([point.BaseDateTime, point.MMSI, point.SOG, point.LON, point.LAT,
                         point.COG, 0, 0, 0])
        for index in range(1, len(trajectory.points)):
            point = trajectory.points[index]
            next_point = trajectory.points[index - 1]
            writer.writerow([point.BaseDateTime, point.MMSI, point.SOG, point.LON, point.LAT,
                             point.COG, (point.BaseDateTime - next_point.BaseDateTime).total_seconds(),
                             point.LON - next_point.LON, point.LAT - next_point.LAT])
