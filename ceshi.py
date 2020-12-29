import math

AB = [1,-3,5,-1]
CD = [4,1,4.5,4.5]
EF = [2,5,-2,6]
PQ = [-3,-4,1,-6]

def angle(v1, v2):
  dx1 = v1[2] - v1[0]
  dy1 = v1[3] - v1[1]
  dx2 = v2[2] - v2[0]
  dy2 = v2[3] - v2[1]
  angle1 = math.atan2(dy1, dx1)
  angle1 = int(angle1 * 180/math.pi)
  # print(angle1)
  angle2 = math.atan2(dy2, dx2)
  angle2 = int(angle2 * 180/math.pi)
  # print(angle2)
  if angle1*angle2 >= 0:
    included_angle = abs(angle1-angle2)
  else:
    included_angle = abs(angle1) + abs(angle2)
    if included_angle > 180:
      included_angle = 360 - included_angle
  return included_angle

ang1 = angle(AB, CD)
print("AB和CD的夹角")
print(ang1)
ang2 = angle(AB, EF)
print("AB和EF的夹角")
print(ang2)
ang3 = angle(AB, PQ)
print("AB和PQ的夹角")
print(ang3)
ang4 = angle(CD, EF)
print("CD和EF的夹角")
print(ang4)
ang5 = angle(CD, PQ)
print("CD和PQ的夹角")
print(ang5)
ang6 = angle(EF, PQ)
print("EF和PQ的夹角")
print(ang6)
