import datetime
str1 = '2021-01-01 23:58:00'
str2 = '2021-01-01 23:59:00'
str3 = '2021-02-02 00:01:00'

dat1 = datetime.datetime.strptime(str1,'%Y-%m-%d %H:%M:%S')
dat2 = datetime.datetime.strptime(str2,'%Y-%m-%d %H:%M:%S')
dat3 = datetime.datetime.strptime(str3,'%Y-%m-%d %H:%M:%S')

print((dat2-dat1).total_seconds())
print((dat3-dat1).total_seconds())