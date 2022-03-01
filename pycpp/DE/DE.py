# -*- coding: cp936 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
import random


# Rastrigr ����
def object_function(x):
    c = 0
    for i in range(0, len(x) - 1):
        c += (100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2)
    s = c
    return s


# ����
def initpara():
    NP = 100  # ��Ⱥ����
    F = 0.6  # ��������
    CR = 0.7  # �������
    generation = 5000  # �Ŵ�����
    len_x = 30
    value_up_range = 30
    value_down_range = -30
    return NP, F, CR, generation, len_x, value_up_range, value_down_range


# ��Ⱥ��ʼ��
def initialtion(NP,len_x,value_down_range,value_up_range):
    np_list = []  # ��Ⱥ��Ⱦɫ��
    for i in range(0, NP):
        x_list = []  # ���壬����
        for j in range(0, len_x):
            x_list.append(value_down_range + random.random() * (value_up_range - value_down_range))
        np_list.append(x_list)
    return np_list


# �б����
def substract(a_list, b_list):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        new_list.append(a_list[i] - b_list[i])
    return new_list


# �б����
def add(a_list, b_list):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        new_list.append(a_list[i] + b_list[i])
    return new_list


# �б������
def multiply(a, b_list):
    b = len(b_list)
    new_list = []
    for i in range(0, b):
        new_list.append(a * b_list[i])
    return new_list


# ����
#�������ȷ���Ŵ���������ԣ���ֹ����ֲ���
def mutation(np_list,NP):
    v_list = []
    for i in range(0, NP):
        r1 = random.randint(0, NP - 1)
        while r1 == i:     #r1���ܵ���i�����ܵ���i��ԭ���Ƿ�ֹ֮����еĽ���������������������Ľ��
            r1 = random.randint(0, NP - 1)
        r2 = random.randint(0, NP - 1)
        while r2 == r1 | r2 == i:
            r2 = random.randint(0, NP - 1)
        r3 = random.randint(0, NP - 1)
        while r3 == r2 | r3 == r1 | r3 == i:
            r3 = random.randint(0, NP - 1)
        #��DE�г����Ĳ�ֲ��������ѡȡ��Ⱥ�е�������ͬ�ĸ��壬�������������ź�������������������ϳ�
        #FΪ��������FԽС���㷨�Ծֲ��������������ã�FԽ���㷨Խ�������ֲ���С�㣬���������ٶȻ���������⣬F��Ӱ����Ⱥ�Ķ����ԡ�
        v_list.append(add(np_list[r1], multiply(F, substract(np_list[r2], np_list[r3]))))
        # if i==0:
        #     print(f'r1:{r1};r2:{r2};r3:{r3}')
        #     print(v_list[0])
    return v_list


# ����
def crossover(np_list, v_list,NP,len_x,CR):
    u_list = []
    for i in range(0, NP):
        vv_list = []
        for j in range(0, len_x):   #len_x=10
            if (random.random() <= CR) | (j == random.randint(0, len_x - 1)):
                #(j == random.randint(0, len_x - 1)��Ϊ��ʹ�����м���������һ�������Ŵ�����һ��
                vv_list.append(v_list[i][j])
            else:
                vv_list.append(np_list[i][j])
        u_list.append(vv_list)
    return u_list
        #CR��Ҫ��ӳ�����ڽ���Ĺ����У��Ӵ��븸�����м������֮�佻����Ϣ���Ĵ�С�̶ȡ�CR��ֵԽ����Ϣ�������ĳ̶�Խ��
        #��֮�����CR��ֵƫС������ʹ��Ⱥ�Ķ����Կ��ټ�С��������ȫ��Ѱ�š�

# ѡ��
def selection(u_list, np_list,NP):
    for i in range(0, NP):
        if object_function(u_list[i]) <= object_function(np_list[i]):
            np_list[i] = u_list[i]
        else:
            np_list[i] = np_list[i]
    return np_list


def DE():
    NP, F, CR, generation, len_x, value_up_range, value_down_range = initpara()   #��ʼ������
    np_list = initialtion(NP,len_x,value_down_range,value_up_range)
    print(f'��ʼ����Ⱥ{np_list}') #��ʼ����Ⱥ
    min_x = []
    min_f = []
    xx = []
    for i in range(0, NP):
        xx.append(object_function(np_list[i]))    #������Ӧ��
    # print(f'��Ӧ��{xx}')
    min_f.append(min(xx))
    min_x.append(np_list[xx.index(min(xx))])
    for i in range(0, generation):   #generation=2000
        v_list = mutation(np_list,NP)    #����
        u_list = crossover(np_list, v_list,NP,len_x,CR)    #����
        np_list = selection(u_list, np_list,NP)    #ѡ��
        xx = []
        for i in range(0, NP):
            xx.append(object_function(np_list[i]))
        min_f.append(min(xx))
        min_x.append(np_list[xx.index(min(xx))])
    # ���
    min_ff = min(min_f)
    print(min_ff)
    return min_f
# ��ͼ
x_label = np.arange(0, generation + 1, 1)
plt.plot(x_label, min_f, color='blue')
plt.title('DE')
plt.xlabel('iteration')
plt.ylabel('fx')
plt.savefig('./iteration-f.png')
plt.show()