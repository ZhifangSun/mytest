"""
类别: 算法
名称: 基于退火算子和差分算子的灰狼优化算法
作者: 孙质方
邮件: zf_sun@hnist.edu.cn
日期: 2021年12月26日
说明:
"""
import random
import numpy
import math
import matplotlib.pyplot as plt
from time import *
import tracemalloc
import numpy as np

# 参数
def initpara():
    F = 0.6  # 缩放因子
    CR = 0.7  # 交叉概率
    return F, CR


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
def substract(a_list, b_list,lb,ub):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        temp=a_list[i] - b_list[i]
        if temp < lb:
            temp = lb
        elif temp > ub:
            temp = ub
        new_list.append(temp)
    return new_list


# 列表相加
def add(a_list, b_list,lb,ub):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        temp = a_list[i] + b_list[i]
        if temp < lb:
            temp = lb
        elif temp > ub:
            temp = ub
        new_list.append(temp)
    return new_list


# 列表的数乘
def multiply(a, b_list,lb,ub):
    b = len(b_list)
    new_list = []
    for i in range(0, b):
        temp = a * b_list[i]
        if temp < lb:
            temp = lb
        elif temp > ub:
            temp = ub
        new_list.append(temp)
    return new_list


# 变异
#变异可以确保遗传基因多样性，防止陷入局部解
def mutation(np_list,NP,F,ub,lb):
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
        v_list.append(add(np_list[r1], multiply(F, substract(np_list[r2], np_list[r3],lb,ub),lb,ub),lb,ub))
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
def selection(u_list, np_list,NP,len_x,object_function):
    for i in range(0, NP):
        if object_function(numpy.array(u_list[i])) <= object_function(np_list[i]):
            for j in range(len_x):
                np_list[i][j] = u_list[i][j]
    return np_list


def DE(np_list,object_function,generation,NP, len_x, value_up_range, value_down_range):
    F, CR= initpara()   #初始化参数
    # np_list = initialtion(NP,len_x,value_down_range,value_up_range)
    min_x = []
    min_f = []
    xx = []
    for i in range(0, NP):
        xx.append(object_function(np_list[i]))   #计算适应度
    # print(f'适应度{xx}')
    min_f.append(min(xx))
    min_x.append(np_list[xx.index(min(xx))])
    for i in range(0, generation):   #generation=2000
        v_list = mutation(np_list,NP,F, value_up_range, value_down_range)    #变异
        u_list = crossover(np_list, v_list,NP,len_x,CR)    #交叉
        np_list = selection(u_list, np_list,NP,len_x,object_function)    #选择
        xx = []
        for i in range(0, NP):
            xx.append(object_function(np_list[i]))
        min_f.append(min(xx))
        min_x.append(np_list[xx.index(min(xx))])
    # 输出
    # print(min_f)
    # min_ff = min(min_f)
    # print(min_ff)
    return min_f

#粒子群
def velocity_update(V, X, pbest, gbest, c1, c2, w, max_val):
    """
    根据速度更新公式更新每个粒子的速度
    :param V: 粒子当前的速度矩阵，20*2 的矩阵
    :param X: 粒子当前的位置矩阵，20*2 的矩阵
    :param pbest: 每个粒子历史最优位置，20*2 的矩阵
    :param gbest: 种群历史最优位置，1*2 的矩阵
    """
    size = X.shape[0]  #size=20
    r1 = np.random.random((size, 1))  #生成20行一列的（0,1）随机数
    r2 = np.random.random((size, 1))
    V = w*V+c1*r1*(pbest-X)+c2*r2*(gbest-X)
    # 防止越界处理
    V[V < -max_val] = -max_val
    V[V > max_val] = max_val
    return V



def position_update(X, V):
    """
    根据公式更新粒子的位置
    :param X: 粒子当前的位置矩阵，维度是 20*2
    :param V: 粒子当前的速度举着，维度是 20*2
    """
    return X+V


def pos(Positions,Size,Dim,Max_iter,fitness_func):
    w = 1
    c1 = 2
    c2 = 2
    r1 = None
    r2 = None
    dim = Dim
    size = Size
    iter_num = Max_iter
    max_val = (ub-lb)*0.15
    best_fitness = float(9e10)
    fitness_val_list = []
    # 初始化种群各个粒子的位置
    X = Positions
    # 初始化各个粒子的速度
    V = np.random.uniform(-(ub-lb)*0.15, (ub-lb)*0.15, size=(size, dim))
    p_fitness=[]
    for i in range(size):
        p_fitness.append(fitness_func(X[i]))  #计算粒子的的适应度值，也就是目标函数值，X 的维度是 size * 2
    g_fitness = min(p_fitness)  #找到所有随机点里的最小值
    fitness_val_list.append(g_fitness)  #压入数组

    # 初始化的个体最优位置和种群最优位置
    pbest = X
    gbest = X[p_fitness.index(g_fitness)]   #找到随机点中自适应度最小的在点坐标
    # 迭代计算
    for i in range(1, iter_num):
        V = velocity_update(V, X, pbest, gbest, c1, c2, w, max_val)
        X = position_update(X, V)
        p_fitness2 = []
        for i in range(size):
            p_fitness2.append(fitness_func(X[i]))  #计算运动后粒子的的适应度值，也就是目标函数值，X 的维度是 size * 2
        g_fitness2 = min(p_fitness2)   #找到运动后所有随机点里的最小值

        # 更新每个粒子的历史最优位置
        for j in range(size):
            if p_fitness[j] > p_fitness2[j]:
                pbest[j] = X[j]
                p_fitness[j] = p_fitness2[j]
            # 更新群体的最优位置
            if g_fitness > g_fitness2:
                gbest = X[p_fitness2.index(g_fitness2)]#找到随机点中自适应度最小的在点坐标
                g_fitness = g_fitness2
            # 记录最优迭代记录
        fitness_val_list.append(g_fitness)
        i += 1

    # 输出迭代结果
    # print(fitness_val_list)
    # print("最优值是：%.5f" %  fitness_val_list[-1])
    # print("最优解是：x=%.5f,y=%.5f" % (gbest[0], gbest[1]))
    return fitness_val_list

#灰狼
def init():
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):  # 形成SearchAgents_no*30个数[-100，100)以内
        Positions[:, i] = numpy.random.uniform(0, 1, SearchAgents_no) * (ub - lb) + lb
    return Positions

def re_gene(Positions,objf,lenth):
    init_Positions = numpy.zeros((SearchAgents_no +lenth, dim))
    # positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):  # 形成SearchAgents_no*30个数[-100，100)以内
        init_Positions[:SearchAgents_no, i] = Positions[:, i]
        for j in range(lenth):
            if init_Positions[j][i]>=(lb + ub)/2:
                init_Positions[SearchAgents_no + j][i] = (lb + ub)/2 - (init_Positions[j][i]-(lb + ub)/2)
            else:
                init_Positions[SearchAgents_no + j][i] = (lb + ub) / 2 + ((lb + ub) / 2-init_Positions[j][i] )
    init_Positions=numpy.array(sorted(init_Positions,key=lambda x:objf(x)))
    for i in range(SearchAgents_no):
        Positions[i, :] = init_Positions[i,:]
    return Positions

def DE_GWO_AS(Positions,objf, lb, ub, dim, SearchAgents_no, Max_iter):
    #F = 0.6  # 缩放因子
    T = 1
    T0 = 1
    K = 0.1
    EPS = 0.1
    ccc = []
    ddd = []
    for i in Positions:
        ccc.append(i[0])
        ddd.append(i[1])
    print(f'c={ccc};')
    print(f'd={ddd};')
    Positions=re_gene(Positions,objf,SearchAgents_no)
    Convergence_curve_1 = []
    best_ans=objf(Positions[0])
    #迭代寻优
    l=0
    while T > EPS:
        # Positions=numpy.array(sorted(Positions,key=lambda x:objf(x)))
        Alpha_score=objf(Positions[0])
        Alpha_pos =Positions[0]
        Beta_pos =Positions[1]
        Delta_pos =Positions[2]
        # 以上的循环里，Alpha、Beta、Delta
        Gamma=math.log(EPS/T0,K)
        a = 2 - l * ((2) / Gamma)  #   a从2线性减少到0
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1]主要生成一个0-1的随机浮点数。
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a;  #  (-a.a)
                C1 = 2 * r2;  #  (0.2)
                # D_alpha表示候选狼与Alpha狼的距离
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j]);  # abs() 函数返回数字的绝对值。Alpha_pos[j]表示Alpha位置，Positions[i,j])候选灰狼所在位置
                X1 = Alpha_pos[j] - A1 * D_alpha;  # X1表示根据alpha得出的下一代灰狼位置向量
                #if i==0 and l==0:
                    #print(f'A1:{A1};C1:{C1};D_alpha:{D_alpha};X1:{X1}')

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a;  #
                C2 = 2 * r2;

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                X2 = Beta_pos[j] - A2 * D_beta;
                #if i==0 and l==0:
                    #print(f'A2:{A2};C2:{C2};D_beta:{D_beta};X2:{X2}')

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                X3 = Delta_pos[j] - A3 * D_delta;
                # temp=((2 / 3) - (l / Gamma)) * X1 + (1 / 3) * X2 + (l / Gamma) * X3
                temp = (2 / 3)*(l / Gamma) * X1 + (1 / 3) * X2 + ((2 / 3) - (2 / 3)*(l / Gamma)) * X3
                if temp<lb:
                    temp=lb
                elif temp>ub:
                    temp=ub
                Positions[i, j] = temp
                #Positions[i, j] = (X1 + X2 + X3) / 3  # 候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。
        #差分进化算子
        r1 = random.randint(0, SearchAgents_no - 1)
        r2 = random.randint(0, SearchAgents_no - 1)
        while r2 == r1:
            r2 = random.randint(0, SearchAgents_no - 1)
        r3 = random.randint(0, SearchAgents_no - 1)
        while r3 == r2 | r3 == r1:
            r3 = random.randint(0, SearchAgents_no - 1)
        # 在DE中常见的差分策略是随机选取种群中的两个不同的个体，将其向量差缩放后与待变异个体进行向量合成
        # F为缩放因子F越小，算法对局部的搜索能力更好，F越大算法越能跳出局部极小点，但是收敛速度会变慢。此外，F还影响种群的多样性。
        F=2*(T0-T)/(T0-EPS)
        v_list = add(Positions[r1], multiply(F, substract(Positions[r2], Positions[r3],lb,ub),lb,ub),lb,ub)
        v_list_ans = objf(numpy.array(v_list))
        Positions=numpy.array(sorted(Positions, key=lambda x: objf(x)))
        ant = 0
        for key in range(SearchAgents_no):
            if objf(Positions[key]) > v_list_ans:
                ant += 1
        # p=1/(1+math.exp((-ant)*T))#退火算子
        p = math.exp(((ant / SearchAgents_no) - 1))
        T*=K
        l+=1
        if random.random() <= p:
            for j in range(dim):
                Positions[len(Positions)-1,j]=v_list[j]
        re_lenth = SearchAgents_no-SearchAgents_no * (Gamma - l) / Gamma  # re_lenth从pop_size线性减小到1
        # print(re_lenth)
        if re_lenth < 1:
            re_lenth = 1
        Positions=re_gene(Positions,objf,math.ceil(re_lenth))
        Alpha_score = objf(Positions[0])
        if Alpha_score<best_ans:
            best_ans=Alpha_score
        Convergence_curve_1.append(best_ans)
    # print(Positions[0])
    aaa=[]
    bbb=[]
    for i in Positions:
        aaa.append(i[0])
        bbb.append(i[1])
    print(f'a={aaa};')
    print(f'b={bbb};')
    return Convergence_curve_1

def GWO(Positions,objf, lb, ub, dim, SearchAgents_no, Max_iter):
    Convergence_curve_2 = []
    #迭代寻优
    for l in range(0, Max_iter):  # 迭代1000
        Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))
        Alpha_score = objf(Positions[0])
        Alpha_pos = list(Positions[0])
        Beta_pos = list(Positions[1])
        Delta_pos = list(Positions[2])
        a = 2 - l * ((2) / Max_iter);  #   a从2线性减少到0

        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1]主要生成一个0-1的随机浮点数。
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a;  #  (-a.a)
                C1 = 2 * r2;  #  (0.2)
                # D_alpha表示候选狼与Alpha狼的距离
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j]);  # abs() 函数返回数字的绝对值。Alpha_pos[j]表示Alpha位置，Positions[i,j])候选灰狼所在位置
                X1 = Alpha_pos[j] - A1 * D_alpha;  # X1表示根据alpha得出的下一代灰狼位置向量
                #if i==0 and l==0:
                    #print(f'A1:{A1};C1:{C1};D_alpha:{D_alpha};X1:{X1}')

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a;  #
                C2 = 2 * r2;

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                X2 = Beta_pos[j] - A2 * D_beta;
                #if i==0 and l==0:
                    #print(f'A2:{A2};C2:{C2};D_beta:{D_beta};X2:{X2}')

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                X3 = Delta_pos[j] - A3 * D_delta;
                temp = (X1 + X2 + X3) / 3
                if temp < lb:
                    temp = lb
                elif temp > ub:
                    temp = ub
                Positions[i, j] = temp
                # 候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。

        Convergence_curve_2.append(Alpha_score)
    return Convergence_curve_2

def vm_GWO(Positions,objf, lb, ub, dim, SearchAgents_no, Max_iter):
    Convergence_curve_2 = []
    #迭代寻优
    for l in range(0, Max_iter):  # 迭代1000
        Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))
        Alpha_score = objf(Positions[0])
        Alpha_pos = list(Positions[0])
        Beta_pos = list(Positions[1])
        Delta_pos = list(Positions[2])
        # a = 2 - l * ((2) / Max_iter);  #   a从2线性减少到0
        a = 1.6*math.exp(-l/Max_iter)
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1]主要生成一个0-1的随机浮点数。
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a;  #  (-a.a)
                C1 = 2 * r2;  #  (0.2)
                # D_alpha表示候选狼与Alpha狼的距离
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j]);  # abs() 函数返回数字的绝对值。Alpha_pos[j]表示Alpha位置，Positions[i,j])候选灰狼所在位置
                X1 = Alpha_pos[j] - A1 * D_alpha;  # X1表示根据alpha得出的下一代灰狼位置向量
                #if i==0 and l==0:
                    #print(f'A1:{A1};C1:{C1};D_alpha:{D_alpha};X1:{X1}')

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a;  #
                C2 = 2 * r2;

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                X2 = Beta_pos[j] - A2 * D_beta;
                #if i==0 and l==0:
                    #print(f'A2:{A2};C2:{C2};D_beta:{D_beta};X2:{X2}')

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                X3 = Delta_pos[j] - A3 * D_delta;
                #if i==0 and l==0:
                    #print(f'A3:{A3};C3:{C3};D_delta:{D_delta};X3:{X3}')
                fai=0.5*math.atan(l)
                ceita=(2/math.pi)*math.acos(1/3)*math.atan(l)
                w1=math.cos(ceita)
                w2=0.5*math.sin(ceita)*math.cos(fai)
                temp = w1*X1 + w2*X2 + (1-w1-w2)*X3
                if temp < lb:
                    temp = lb
                elif temp > ub:
                    temp = ub
                Positions[i, j] = temp
                #候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。

        Convergence_curve_2.append(Alpha_score)
    return Convergence_curve_2

#适应度函数
def F7(x):
    ss=numpy.sum(x**2)
    cc=0
    for i in range(len(x)):
        cc+=math.cos(2*math.pi*x[i])
    s=-20*math.exp(-0.2*math.sqrt(ss/len(x)))-math.exp(cc/len(x))+20+math.exp(1)
    return s

def F1(x):#[-100,100]
    ss=numpy.sum(x**2)
    s=ss
    return s

def F2(x):#[-100,100]
    cc=0
    c=1
    for i in range(len(x)):
        cc+=abs(x[i])
        c*=abs(x[i])
    s=cc+c
    return s

def F3(x):#[-100,100]
    cc=0
    for i in range(1,len(x)+1):
        c=0
        for j in range(0,i):
            c+=x[j]
        cc+=c**2
    s=cc
    return s

def F4(x):#[-100,100]
    cc=-99999999
    for i in range(0,len(x)):
        if abs(x[i])>cc:
            cc=abs(x[i])
    s=cc
    return s

def F5(x):#[-100,100]
    ss = numpy.sum(x ** 2)
    s = ss**2
    return s

def F6(x):#[-100,100]
    ss = 0
    for i in range(len(x)):
        ss+=abs(x[i])
    s = ss
    return s

def F11(x):#[-1,1]
    cc = 0
    for i in range(0, len(x)):
        cc += (x[i] ** 2)*(2+math.sin(1/x[i]))
    s = cc
    return s

def F12(x):#[-1,1]
    cc=0
    ss=numpy.sum(x**2)
    s=-math.exp(-0.5*ss)
    return s

def F8(x):#[-100,100]
    ss = numpy.sum(x**2/4000)+1
    c=1
    for i in range(1,len(x)+1):
        c*=math.cos(x[i-1]/math.sqrt(i))
    s = ss-c
    return s

def F9(x):#[-100,100]
    ss = numpy.sum(x**2)
    s = 1-math.cos(2*math.pi*math.sqrt(ss))+0.1*math.sqrt(ss)
    return s

def F10(x):#[-5,10]
    ss = numpy.sum(x**2)
    c=0
    for i in range(1, len(x) + 1):
        c += 0.5*i*x[i-1]
    s=ss+c**2+c**4
    return s

def F13(x):#[-5,10]
    c=0
    for i in range(0, len(x)):
        c += -x[i]*math.sin(math.sqrt(abs(x[i])))
    s=c
    return s

def F14(x):#[-5,10]
    c=0
    for i in range(0, len(x)-1):
        c += (100*(x[i+1]-x[i]**2)**2+(x[i]-1)**2)
    s=c
    return s

tracemalloc.start()
#主程序
func_details = ['F1', -100, 100, 30]
function_name = func_details[0]
Max_iter = 10#迭代次数
lb = -100#下界-10000000000000000000000
ub = 100#上届10000000000000000000000
dim = 2#狼的寻值范围30
SearchAgents_no = 100#寻值的狼的数量
Fx=F8
positions=init()
begin_time = time()
X=[]
# for i in range(30):
positions_1=positions.copy()
x = DE_GWO_AS(positions_1,Fx, lb, ub, dim, SearchAgents_no, Max_iter)#改进灰狼
    # X.append(x[len(x)-1])
    # print(x)
# print(f'改进灰狼平均值{np.mean(X)}标准差{np.std(X,ddof=1)}')
# end_time = time()
# run_time = end_time - begin_time
# print(run_time)
# begin_time = time()
# Y=[]
# for i in range(30):
#     positions_2 = positions.copy()
#     y = GWO(positions_2,Fx, lb, ub, dim, SearchAgents_no, Max_iter)#经典灰狼
#     Y.append(y[len(y)-1])
#     print(y)
# # print(f'传统灰狼平均值{np.mean(Y)}标准差{np.std(Y,ddof=1)}')
#
# end_time = time()
# run_time = end_time - begin_time
# print(run_time)
# Z=[]
# for i in range(30):
#     positions_3 = positions.copy()
#     z=pos(positions_3,SearchAgents_no,dim,Max_iter,Fx)
#     Z.append(z[len(z)-1])
#     print(z)
# # print(f'粒子群平均值{np.mean(Z)}标准差{np.std(Z,ddof=1)}')
#
# D=[]
# for i in range(30):
#     positions_4 = positions.copy()
#     de=DE(positions_4,Fx,Max_iter,SearchAgents_no,dim,ub,lb)
#     D.append(de[len(de)-1])
#     print(de)
# # print(f'差分进化平均值{np.mean(D)}标准差{np.std(D,ddof=1)}')
#
# VM=[]
# for i in range(30):
#     positions_5 = positions.copy()
#     vm = vm_GWO(positions_5,Fx, lb, ub, dim, SearchAgents_no, Max_iter)
#     VM.append(vm[len(vm)-1])
#     print(vm)
# # print(f'变权灰狼平均值{np.mean(VM)}标准差{np.std(VM,ddof=1)}')
#
#
# current, peak = tracemalloc.get_traced_memory()
# print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
# tracemalloc.stop()
#
# # 画图
# x_label = numpy.arange(0, len(x), 1)
# y_label = numpy.arange(0, len(y), 1)
# z_label = numpy.arange(0, len(z), 1)
# de_label = numpy.arange(0, len(de), 1)
# vm_label = numpy.arange(0, len(vm), 1)
# plt.plot(x_label[0:], x[0:], color='blue')
# plt.plot(y_label[0:], y[0:], color='red')
# plt.plot(z_label[0:], z[0:], color='green')
# plt.plot(de_label[0:], de[0:], color='black')
# plt.plot(vm_label[0:], vm[0:], color='yellow')
# plt.title('DE_GWO_AS')
# plt.xlabel('iteration')
# plt.ylabel('fx')
# plt.legend(('ASDE_GWO', 'GWO','PSO','DE','vm_GWO'))
# plt.savefig('./iteration-f.png')
# plt.show()
# # print(f'迭代末端值比较：改进灰狼{x[min(len(x)-1,len(y)-1,len(z)-1,len(de)-1)]}经典灰狼{y[min(len(x)-1,len(y)-1,len(z)-1,len(de)-1)]}粒子群{z[min(len(x)-1,len(y)-1,len(z)-1,len(de)-1)]}差分进化{de[min(len(x)-1,len(y)-1,len(z)-1,len(de)-1)]}')
# print(f'迭代末端值比较：改进灰狼{x[300]}经典灰狼{y[len(y)-1]}粒子群{z[len(z)-1]}差分进化{de[len(de)-1]}变权灰狼{vm[len(vm)-1]}')