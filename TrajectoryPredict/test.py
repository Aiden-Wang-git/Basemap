import matplotlib.pyplot as plt

x_values = [1000000, 2000000, 3000000]
y_values = [1, 2, 3]
plt.plot(x_values, y_values)
plt.ticklabel_format(axis="x", style='plain', scilimits=(0,0))
plt.show()