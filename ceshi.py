import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d




x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x**2/9.0)
f1 = interp1d(x, y, kind='nearest')
f2 = interp1d(x, y, kind='zero')
f3 = interp1d(x, y, kind='quadratic')

xnew = np.linspace(0, 10, num=1001, endpoint=True)
plt.plot(x, y, 'o')
plt.plot(xnew, f1(xnew), '-', xnew, f2(xnew), '--', xnew, f3(xnew), ':')
plt.legend(['data', 'nearest', 'zero', 'quadratic'], loc='best')
plt.show()
