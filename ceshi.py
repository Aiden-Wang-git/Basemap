
lst=[1,3,5,3,4,4,2,9,6,7]
set_lst=set(lst)
#set会生成一个元素无序且不重复的可迭代对象，也就是我们常说的去重
print(lst)
print(set_lst)
if len(set_lst)==len(lst):
    print('列表里的元素互不重复！')
else:
    print('列表里有重复的元素！')
