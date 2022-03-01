import random
import numpy
import math
import matplotlib.pyplot as plt
from time import *
import tracemalloc


def GWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):


    # 初始化 alpha, beta, and delta_pos
    Alpha_pos = numpy.zeros(dim)  # 位置.形成30的列表
    Alpha_score = float("inf")  # 这个是表示“正负无穷”,所有数都比 +inf 小；正无穷：float("inf"); 负无穷：float("-inf")

    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = numpy.zeros(dim)
    Delta_score = float("inf")  # float() 函数用于将整数和字符串转换成浮点数。

    # list列表类型
    if not isinstance(lb, list):  # 作用：来判断一个对象是否是一个已知的类型。 其第一个参数（object）为对象，第二个参数（type）为类型名，若对象的类型与参数二的类型相同则返回True
        lb = [lb] * dim  # 生成[-100，-100，.....-100]30个
    if not isinstance(ub, list):
        ub = [ub] * dim  # 生成[100，100，.....100]30个

    # Initialize the positions of search agents初始化所有狼的位置
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):  # 形成5*30个数[-100，100)以内
        Positions[:, i] = numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]  # 形成[5个0-1的数]*（100-（-100））-100
    Convergence_curve = []
    #print(f'输出所有狼的位置向量{Positions}')




    #迭代寻优
    for l in range(0, Max_iter):  # 迭代1000
        ans=[]
        dic={}
        for i in range(0, SearchAgents_no):  # 5
            # 返回超出搜索空间边界的搜索代理

            for j in range(dim):  # 30
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])  # clip这个函数将将数组中的元素限制在a_min(-100), a_max(100)之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。

            # 计算每个搜索代理的目标函数
            fitness = objf(Positions[i, :])  # 把某行数据带入函数计算
            #ans.append(fitness)
            dic[fitness]=Positions[i]
            #print("经过计算得到：",fitness)
            """
        dic{121465.80996341376: array([ 74.84312946, -84.10898145, -89.4842132 ,  90.60461162,-49.41397479,  17.10349583,  14.47680713,  26.53033461,2.18789427,  66.61888375,  87.44940811, -76.85334651,86.5139396 , -19.53948112, -74.7832266 ,  88.42018406,15.01097471, -82.75722383,  98.64034859, -95.75979775,82.08344322,  12.18612299, -27.58632131,  12.54111753,20.80534277,  47.4107642 ,  21.35351692,  87.85279841,-64.58711325, -29.10685545]),
             96570.20938966161: array([-14.05749154,  17.60318562,  61.11728054,  21.00355853,28.18953052,  70.35591729, -66.43602209, -59.26696927,-38.38535383,  30.11136962,  -3.66806539, -14.0570639 ,69.86977691, -96.64330117, -31.5078005 , -30.96194223,-42.82597491, -90.44360717, -82.59846987,  18.41238478,11.60466323, -81.16734092,  83.07360299,  97.00708995,99.63935591, -61.74595688, -62.89314812, -15.61134234,-10.64107501,  34.4666513 ]),
            .......
            """
            """
            if l==0:
                print(f'初始化每一条狼的自适应度{fitness}')

            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Alpha_score = fitness  # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness < Beta_score):
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score):
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()
            """
        if l==0:
            print(f'1111111111{dic}')
        dic=dict(sorted(dic.items(),key=lambda x:x[0],reverse=False))
        Alpha_score=list(dic.keys())[0]
        Alpha_pos =list(dic.values())[0]
        Beta_score=list(dic.keys())[1]
        Beta_pos =list(dic.values())[1]
        Delta_score=list(dic.keys())[2]
        Delta_pos =list(dic.values())[2]
        print(f'输出最优的三个自适应度{Alpha_score}和{Beta_score}和{Delta_score}')
        #print(f'输出最优的三个自适应度{Alpha_pos}和{Beta_pos}和{Delta_pos}')
        # 以上的循环里，Alpha、Beta、Delta

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

        Convergence_curve.append(Alpha_score)
    # 画图
    x_label = numpy.arange(0, Max_iter, 1)
    plt.plot(x_label, Convergence_curve, color='blue')
    plt.title('GWO')
    plt.xlabel('iteration')
    plt.ylabel('fx')
    plt.savefig('./iteration-f.png')
    plt.show()
    print(f'qwert{Convergence_curve[150]}')
        #if l==0:
            #print(f'第一次迭代后的狼群位置向量{Positions}')
        #if (l % 1 == 0):
            #print(['迭代次数为' + str(l) + ' 的迭代结果' + str(Alpha_score)]);  # 每一次的迭代结果

#适应度函数
def F1(x):
    ss=numpy.sum(x**2)
    cc=0
    for i in range(len(x)):
        cc+=math.cos(2*math.pi*x[i])
    s=-20*math.exp(-0.2*math.sqrt(ss/len(x)))-math.exp(cc/len(x))+20+math.exp(1);
    return s




begin_time = time()
tracemalloc.start()
#主程序
func_details = ['F1', -100, 100, 30]
function_name = func_details[0]
Max_iter = 200#迭代次数
lb = -32#下界
ub = 32#上届
dim = 30#狼的寻值范围
SearchAgents_no = 100#寻值的狼的数量
x = GWO(F1, lb, ub, dim, SearchAgents_no, Max_iter)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
tracemalloc.stop()
end_time = time()
run_time = end_time - begin_time
print(run_time)