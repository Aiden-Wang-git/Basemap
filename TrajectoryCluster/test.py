# import matplotlib.pyplot as plt
# import numpy as np
#
# list1=[1,2,3,4,5,6,2,3,4,6,7,5,7]
# list2=[2,3,4,5,8,9,2,1,3,4,5,2,4]
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.title('显示中文标题')
# plt.xlabel("横坐标")
# plt.ylabel("纵坐标")
# x=np.arange(0,len(list1))+1
# x[0]=1
# my_x_ticks = np.arange(1, 14, 1)
# plt.xticks(my_x_ticks)
# plt.plot(x,list1,label='list1',color='g',linewidth=0.5,linestyle=':')#添加linestyle设置线条类型
# plt.plot(x,list2,label='list2',color='b',linewidth=0.1,linestyle='-')
# plt.legend()
# plt.grid()#添加网格
# plt.show()

x = 1
y = 2
z = 1
x+=y
print(x)
a =b =c= d= 1
a,b = c,d
