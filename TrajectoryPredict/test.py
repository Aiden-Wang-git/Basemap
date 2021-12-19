import copy
dic1 = {'1':[1],'2':[2,2]}
dic2 = copy.deepcopy(dic1)
dic1['1'].append(1)
print(dic1)
print(dic2)