from __future__ import division
import time
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)

#############################################################################
# 生成随机数据
X = 5 * rng.rand(10000, 1)
y = np.sin(X).ravel()

# 在标签中对每50个结果标签添加噪声

y[::50] += 2 * (0.5 - rng.rand(int(X.shape[0] / 50)))

X_plot = np.linspace(0, 5, 100000)[:, None]

#############################################################################
# 训练SVR模型

# 训练规模
train_size = 100
# 初始化SVR
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})
# 记录训练时间
t0 = time.time()
# 训练
svr.fit(X[:train_size], y[:train_size])
svr_fit = time.time() - t0

t0 = time.time()
# 测试
y_svr = svr.predict(X_plot)
svr_predict = time.time() - t0

#############################################################################
# 对结果进行显示
plt.scatter(X[:100], y[:100], c='k', label='data', zorder=1)
# plt.hold('on')
plt.plot(X_plot, y_svr, c='r',
         label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))

plt.xlabel('data')
plt.ylabel('target')
plt.title('SVR versus Kernel Ridge')
plt.legend()
plt.show()

##############################################################################
# 对训练和测试的过程耗时进行可视化
X = 5 * rng.rand(1000000, 1)
y = np.sin(X).ravel()
y[::50] += 2 * (0.5 - rng.rand(int(X.shape[0] / 50)))
sizes = np.logspace(1, 4, 7)
for name, estimator in {
    "SVR": SVR(kernel='rbf', C=1e1, gamma=10)}.items():
    train_time = []
    test_time = []
    for train_test_size in sizes:
        t0 = time.time()
        estimator.fit(X[:int(train_test_size)], y[:int(train_test_size)])
        train_time.append(time.time() - t0)

        t0 = time.time()
        estimator.predict(X_plot[:1000])
        test_time.append(time.time() - t0)

    plt.plot(sizes, train_time, 'o-', color="b" if name == "SVR" else "g",
             label="%s (train)" % name)
    plt.plot(sizes, test_time, 'o--', color="r" if name == "SVR" else "g",
             label="%s (test)" % name)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Train size")
plt.ylabel("Time (seconds)")
plt.title('Execution Time')
plt.legend(loc="best")
plt.show()
##############################################################################
# 对训练和测试的过程耗时进行可视化
X = 5 * rng.rand(1000000, 1)
y = np.sin(X).ravel()
y[::50] += 2 * (0.5 - rng.rand(int(X.shape[0] / 50)))
sizes = np.logspace(1, 4, 7)
for name, estimator in {
    "SVR": SVR(kernel='rbf', C=1e1, gamma=10)}.items():
    train_time = []
    test_time = []
    for train_test_size in sizes:
        t0 = time.time()
        estimator.fit(X[:int(train_test_size)], y[:int(train_test_size)])
        train_time.append(time.time() - t0)

        t0 = time.time()
        estimator.predict(X_plot[:1000])
        test_time.append(time.time() - t0)

    plt.plot(sizes, train_time, 'o-', color="b" if name == "SVR" else "g",
             label="%s (train)" % name)
    plt.plot(sizes, test_time, 'o--', color="r" if name == "SVR" else "g",
             label="%s (test)" % name)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Train size")
plt.ylabel("Time (seconds)")
plt.title('Execution Time')
plt.legend(loc="best")
plt.show()