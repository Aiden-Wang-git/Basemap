dic1 = {'1':'a','2':'b'}
dic2 = dic1.copy()
dic1['1'] = 'aa'
print(dic1)
print(dic2)