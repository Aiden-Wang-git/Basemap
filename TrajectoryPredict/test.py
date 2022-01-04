import random

map = {1: '张三', 2: '李四', 3: '王五', 4: '赵六', 5: '王麻子', 6: '包子', 7: '豆浆'}

keys = random.sample(map.keys(), 4)
for key in keys:
    print(key, map[key])