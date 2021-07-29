# 豪斯多夫python实现过程
import numpy as np
import operator


def hausdorff(t1, t2):
    A = []
    B = []
    for t in t1:
        A.append([t.LAT,t.LON])
    for t in t2:
        B.append([t.LAT,t.LON])
    asize = [len(A), len(A[0])]
    bsize = [len(B), len(B[0])]
    fhd = 0
    for i in range(asize[0]):
        mindist = float("inf")
        for j in range(bsize[0]):
            tempdist = np.linalg.norm(list(map(operator.sub, A[i], B[j])))
            if tempdist < mindist:
                mindist = tempdist
        fhd = fhd + mindist
    fhd = fhd / asize[0]
    rhd = 0
    for j in range(bsize[0]):
        mindist = float("inf")
        for i in range(asize[0]):
            tempdist = np.linalg.norm(list(map(operator.sub, A[i], B[j])))
            if tempdist < mindist:
                mindist = tempdist
        rhd = rhd + mindist
    rhd = rhd / bsize[0]
    mhd = max(fhd, rhd)
    return mhd

