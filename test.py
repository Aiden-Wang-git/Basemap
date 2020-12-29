import numpy as np
a = np.array([[0,2,3],[0,0,4],[0,0,0]])
b = [3,4]
c = [5,6]
a = np.triu(a)
a+=a.T-np.diag(a.diagonal())
print(a)