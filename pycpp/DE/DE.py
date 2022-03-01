# -*- coding: cp936 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
import random


# Rastrigr 函数
def object_function(x):
    c = 0
    for i in range(0, len(x) - 1):
        c += (100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2)
    s = c
    return s


# 参数
def initpara():
    NP = 100  # 种群数量
    F = 0.6  # 缩放因子
    CR = 0.7  # 交叉概率
    generation = 5000  # 遗传代数
    len_x = 30
    value_up_range = 30
    value_down_range = -30
    return NP, F, CR, generation, len_x, value_up_range, value_down_range


# 种群初始化
def initialtion(NP,len_x,value_down_range,value_up_range):
    np_list = []  # 种群，染色体
    for i in range(0, NP):
        x_list = []  # 个体，基因
        for j in range(0, len_x):
            x_list.append(value_down_range + random.random() * (value_up_range - value_down_range))
        np_list.append(x_list)
    return np_list


# 列表相减
def substract(a_list, b_list):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        new_list.append(a_list[i] - b_list[i])
    return new_list


# 列表相加
def add(a_list, b_list):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        new_list.append(a_list[i] + b_list[i])
    return new_list


# 列表的数乘
def multiply(a, b_list):
    b = len(b_list)
    new_list = []
    for i in range(0, b):
        new_list.append(a * b_list[i])
    return new_list


# 变异
#变异可以确保遗传基因多样性，防止陷入局部解
def mutation(np_list,NP):
    v_list = []
    for i in range(0, NP):
        r1 = random.randint(0, NP - 1)
        while r1 == i:     #r1不能等于i，不能等于i的原因是防止之后进行的交叉操作出现自身和自身交叉的结果
            r1 = random.randint(0, NP - 1)
        r2 = random.randint(0, NP - 1)
        while r2 == r1 | r2 == i:
            r2 = random.randint(0, NP - 1)
        r3 = random.randint(0, NP - 1)
        while r3 == r2 | r3 == r1 | r3 == i:
            r3 = random.randint(0, NP - 1)
        #在DE中常见的差分策略是随机选取种群中的两个不同的个体，将其向量差缩放后与待变异个体进行向量合成
        #F为缩放因子F越小，算法对局部的搜索能力更好，F越大算法越能跳出局部极小点，但是收敛速度会变慢。此外，F还影响种群的多样性。
        v_list.append(add(np_list[r1], multiply(F, substract(np_list[r2], np_list[r3]))))
        # if i==0:
        #     print(f'r1:{r1};r2:{r2};r3:{r3}')
        #     print(v_list[0])
    return v_list


# 交叉
def crossover(np_list, v_list,NP,len_x,CR):
    u_list = []
    for i in range(0, NP):
        vv_list = []
        for j in range(0, len_x):   #len_x=10
            if (random.random() <= CR) | (j == random.randint(0, len_x - 1)):
                #(j == random.randint(0, len_x - 1)是为了使变异中间体至少有一个基因遗传给下一代
                vv_list.append(v_list[i][j])
            else:
                vv_list.append(np_list[i][j])
        u_list.append(vv_list)
    return u_list
        #CR主要反映的是在交叉的过程中，子代与父代、中间变异体之间交换信息量的大小程度。CR的值越大，信息量交换的程度越大。
        #反之，如果CR的值偏小，将会使种群的多样性快速减小，不利于全局寻优。

# 选择
def selection(u_list, np_list,NP):
    for i in range(0, NP):
        if object_function(u_list[i]) <= object_function(np_list[i]):
            np_list[i] = u_list[i]
        else:
            np_list[i] = np_list[i]
    return np_list


def DE():
    NP, F, CR, generation, len_x, value_up_range, value_down_range = initpara()   #初始化参数
    np_list = initialtion(NP,len_x,value_down_range,value_up_range)
    print(f'初始化种群{np_list}') #初始化种群
    min_x = []
    min_f = []
    xx = []
    for i in range(0, NP):
        xx.append(object_function(np_list[i]))    #计算适应度
    # print(f'适应度{xx}')
    min_f.append(min(xx))
    min_x.append(np_list[xx.index(min(xx))])
    for i in range(0, generation):   #generation=2000
        v_list = mutation(np_list,NP)    #变异
        u_list = crossover(np_list, v_list,NP,len_x,CR)    #交叉
        np_list = selection(u_list, np_list,NP)    #选择
        xx = []
        for i in range(0, NP):
            xx.append(object_function(np_list[i]))
        min_f.append(min(xx))
        min_x.append(np_list[xx.index(min(xx))])
    # 输出
    min_ff = min(min_f)
    print(min_ff)
    return min_f
# 画图
x_label = np.arange(0, generation + 1, 1)
plt.plot(x_label, min_f, color='blue')
plt.title('DE')
plt.xlabel('iteration')
plt.ylabel('fx')
plt.savefig('./iteration-f.png')
plt.show()