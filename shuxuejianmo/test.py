import matplotlib
import random
import matplotlib.pyplot as plt
from collections import OrderedDict

import xlrd
#打开Excel文件读取数据('文件所在路径')
dataa = xlrd.open_workbook(r'shuju.xlsx')
#根据sheet索引获取sheet内容
sheet = dataa.sheet_by_index(0)
#获取行数
sheet_nrows = sheet.nrows
# 创建存放这列数据的列表
list1 = []
# 从第三行开始读取数据
i = 0
dis = {}
while i < sheet_nrows:
    dis[sheet.cell(i,0).value]=sheet.cell(i,1).value
    i +=1
# new_list = list(set(list1))                # 去重
# new_list.sort(key=list1.index)             # 去重后顺序不变
print(dis)
sorted_pairs = sorted(dis.items(), key=lambda k: abs(k[1]), reverse=True)
ordered_dict = OrderedDict(sorted_pairs)
print(ordered_dict)
city_name = []
data = []
count = 1
for k,v in ordered_dict.items():
    if(count<=20):
        city_name.append(k)
        data.append(v)
        count= count+1
# 中文乱码和坐标轴负号处理。
matplotlib.rc('font', family='SimHei', weight='bold')
plt.rcParams['axes.unicode_minus'] = False


# 绘图。
fig, ax = plt.subplots()
b = ax.barh(range(len(city_name)), data, color='#6699CC')

# 为横向水平的柱图右侧添加数据标签。
for rect in b:
    w = rect.get_width()
    ax.text(w, rect.get_y() + rect.get_height() / 2.0, '%f' %
            w, ha='left', va='center')

# 设置Y轴纵坐标上的刻度线标签。
ax.set_yticks(range(len(city_name)))
ax.set_yticklabels(city_name)
# plt.figure(figsize=(28,3))
# 不要X横坐标上的label标签。
plt.xticks(())

plt.title('水平横向的柱状图', loc='center', fontsize='25',
          fontweight='bold', color='red')

plt.show()