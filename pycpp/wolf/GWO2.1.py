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
    K = 0.97
    EPS = 0.01
    Positions=re_gene(Positions,objf,SearchAgents_no)
    Convergence_curve_1 = []
    Alpha_score = objf(Positions[0])
    Alpha_pos = Positions[0]
    Beta_pos = Positions[1]
    Delta_pos = Positions[2]
    #迭代寻优
    l=0
    while T > EPS:
        # Positions=numpy.array(sorted(Positions,key=lambda x:objf(x)))
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
        Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))
        re_lenth = SearchAgents_no - SearchAgents_no * (Gamma - l) / Gamma  # re_lenth从pop_size线性减小到1
        if re_lenth < 1:
            re_lenth = 1
        jpg=0
        for time in range(math.ceil(re_lenth)):
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
            v_list=add(Positions[r1], multiply(F, substract(Positions[r2], Positions[r3],lb,ub),lb,ub),lb,ub)
            v_list_ans = objf(numpy.array(v_list))
            ant=0
            for key in range(SearchAgents_no):
                if objf(Positions[key]) > v_list_ans:
                    ant += 1
        # p=1/(1+math.exp((-ant)*T))#退火算子
            p = math.exp(((ant / SearchAgents_no) - 1))
            if random.random() <= p:
                # Alpha_pos = Alpha
                # Beta_pos = Beta
                # Delta_pos = Delta
                for i in range(SearchAgents_no-jpg-1,SearchAgents_no):
                    for j in range(dim):
                        Positions[i,j]=v_list[j]
        # else:
        #     Alpha_pos = Positions[0]
        #     Beta_pos = Positions[1]
        #     Delta_pos = Positions[2]
        Positions=re_gene(Positions,objf,math.ceil(re_lenth))
        Alpha_pos = Positions[0]
        Beta_pos = Positions[1]
        Delta_pos = Positions[2]
        Alpha_score=objf(Alpha_pos)
        Convergence_curve_1.append(Alpha_score)
        T*=K
        l+=1
    # print(Positions[0])
    return Convergence_curve_1

def GWO(Positions,objf, lb, ub, dim, SearchAgents_no, Max_iter):
    Convergence_curve_2 = []
    #迭代寻优
    for l in range(0, Max_iter):  # 迭代1000
        for i in range(0, SearchAgents_no):  # 5
            for j in range(dim):  # 30
                Positions[i, j] = numpy.clip(Positions[i, j], lb, ub)  # clip这个函数将将数组中的元素限制在a_min(-100), a_max(100)之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。
        Positions = numpy.array(sorted(Positions, key=lambda x: objf(x)))
        Alpha_score=objf(Positions[0])
        Alpha_pos =list(Positions[0])
        Beta_pos =list(Positions[1])
        Delta_pos =list(Positions[2])
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
                #if i==0 and l==0:
                    #print(f'A3:{A3};C3:{C3};D_delta:{D_delta};X3:{X3}')

                Positions[i, j] = (X1 + X2 + X3) / 3  # 候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。

        Convergence_curve_2.append(Alpha_score)
    return Convergence_curve_2
#适应度函数
def F1(x):
    ss=numpy.sum(x**2)
    cc=0
    for i in range(len(x)):
        cc+=math.cos(2*math.pi*x[i])
    s=-20*math.exp(-0.2*math.sqrt(ss/len(x)))-math.exp(cc/len(x))+20+math.exp(1)
    return s

def F2(x):#[-100,100]
    ss=numpy.sum(x**2)
    s=ss
    return s

def F3(x):#[-100,100]
    cc=0
    c=1
    for i in range(len(x)):
        cc+=math.floor(x[i])
        c*=math.floor(x[i])
    s=cc+c
    return s

def F4(x):#[-100,100]
    cc=0
    for i in range(0,len(x)):
        for j in range(0,i+1):
            cc+=x[j]**2
    s=cc
    return s

def F5(x):#[-100,100]
    cc=-99999999
    for i in range(0,len(x)):
        if math.floor(x[i])>cc:
            cc=x[i]
    s=cc
    return s

def F6(x):#[-100,100]
    ss = numpy.sum(x ** 2)
    s = ss**2
    return s

def F7(x):#[-100,100]
    ss = 0
    for i in range(len(x)):
        ss+=math.floor(x[i])
    s = ss
    return s

def F8(x):#[-1,1]
    cc = 0
    for i in range(0, len(x)):
        cc += (x[i] ** 2)*(2+math.sin(1/x[i]))
    s = cc
    return s

def F9(x):#[-1,1]
    cc=0
    ss=numpy.sum(x**2)
    s=-math.exp(-0.5*ss)
    return s

def F10(x):#[-100,100]
    ss = numpy.sum(x**2/4000)+1
    c=1
    for i in range(1,len(x)+1):
        c*=math.cos(x[i-1]/math.sqrt(i))
    s = ss-c
    return s

def F11(x):#[-100,100]
    ss = numpy.sum(x**2)
    s = 1-math.cos(2*math.pi*math.sqrt(ss))+0.1*math.sqrt(ss)
    return s

def F12(x):#[-5,10]
    ss = numpy.sum(x**2)
    c=0
    for i in range(1, len(x) + 1):
        c += 0.5*i*x[i-1]
    s=ss+c**2+c**4
    return s

def F13(x):#[-500,500]
    c=0
    for i in range(0, len(x)):
        c += -x[i]*math.sin(math.sqrt(abs(x[i])))
    s=c
    return s
tracemalloc.start()
#主程序
func_details = ['F1', -100, 100, 30]
function_name = func_details[0]
Max_iter = 100#迭代次数
lb = -500#下界-10000000000000000000000
ub = 500#上届10000000000000000000000
dim = 30#狼的寻值范围30
SearchAgents_no = 100#寻值的狼的数量
positions_1=init()
positions_2=positions_1.copy()
begin_time = time()
x = DE_GWO_AS(positions_1,F13, lb, ub, dim, SearchAgents_no, Max_iter)#改进灰狼
end_time = time()
run_time = end_time - begin_time
print(run_time)
begin_time = time()
y = GWO(positions_2,F13, lb, ub, dim, SearchAgents_no, Max_iter)#经典灰狼
end_time = time()
run_time = end_time - begin_time
print(run_time)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
tracemalloc.stop()

# 画图
x_label = numpy.arange(0, len(x), 1)
y_label = numpy.arange(0, len(y), 1)
plt.plot(x_label[0:], x[0:], color='blue')
plt.plot(y_label[0:], y[0:], color='red')
plt.title('DE_GWO_AS')
plt.xlabel('iteration')
plt.ylabel('fx')
plt.legend(('ASDE_GWO', 'GWO'))
plt.savefig('./iteration-f.png')
plt.show()
print(f'迭代末端值比较：改进灰狼{x[min(len(x)-1,len(y)-1)]}经典灰狼{y[min(len(x)-1,len(y)-1)]}')