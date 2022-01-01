import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 时间
rng = pd.date_range('2016/01/01',periods=90,freq='D')
print(rng)

# 数据
ts = pd.Series(np.random.randn(len(rng)),index = rng)
print(ts)

# 重采样
print(ts.resample('3D').sum())
print("==========")
print(ts.resample('H').interpolate('linear'))

# 滑动窗口
print("==========")
r = ts.rolling(window=10)
print(r.mean())

# 画图
plt.figure(figsize=(15,5))
ts.plot(style='r--')
ts.rolling(window=10).mean().plot(style='b')
plt.show()