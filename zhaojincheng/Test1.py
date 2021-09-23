for hang in range(1,10):
    # print("第"+str(hang)+"行")
    for lie in range(1,hang+1):
        # print("第"+str(lie),end="列,")
        multiply = hang*lie
        print(str(lie)+"*"+str(hang)+"="+str(multiply),end=" ")
    if hang != 9:
        print()

print("===============")
nums = []
nums.append("昆山")
nums.append("赵瑾程")
nums.append("国际学校")
print(nums)

# for i in range(0,len(nums)):
#     print(nums[i]+"你好")

for xuexiao in nums:
    print(xuexiao + "你好")

print("=====================")

hello = "vga"
print(len(hello))

print("============")
dic = {}
dic['小明'] = 89
dic['赵瑾程'] = 99
dic['皮孩子'] = 59
dic['小白'] = 67
for i in dic:
    print(i+"得了"+str(dic[i])+"分")

print("我们现在要做个算法题了")
a = input("请输入2进制数字a:")
