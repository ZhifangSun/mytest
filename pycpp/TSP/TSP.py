# -*- coding: utf-8 -*-
"""
粒子群算法求解TSP问题
随机在（0,101）二维平面生成20个点
距离最小化
"""
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from time import *
import tracemalloc
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)


def calDistance(CityCoordinates):
    '''
    计算城市间距离
    输入：CityCoordinates-城市坐标；
    输出：城市间距离矩阵-dis_matrix
    '''
    dis_matrix = pd.DataFrame(data=None, columns=range(len(CityCoordinates)), index=range(len(CityCoordinates)))
    for i in range(len(CityCoordinates)):
        xi, yi = CityCoordinates[i][0], CityCoordinates[i][1]
        for j in range(len(CityCoordinates)):
            xj, yj = CityCoordinates[j][0], CityCoordinates[j][1]
            if (xi == xj) & (yi == yj):
                dis_matrix.iloc[i, j] = round(math.pow(10, 10))
            else:
                dis_matrix.iloc[i, j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 2)
    return dis_matrix


def calFitness(line, dis_matrix):
    '''
    计算路径距离，即评价函数
    输入：line-路径，dis_matrix-城市间距离矩阵；
    输出：路径距离-dis_sum
    '''
    dis_sum = 0
    dis = 0
    for i in range(len(line) - 1):
        dis = dis_matrix.loc[line[i], line[i + 1]]  # 计算距离
        dis_sum = dis_sum + dis
    dis = dis_matrix.loc[line[-1], line[0]]
    dis_sum = dis_sum + dis
    return round(dis_sum, 1)


def draw_path(line, CityCoordinates):
    '''
    #画路径图
    输入：line-路径，CityCoordinates-城市坐标；
    输出：路径图
    '''
    x, y = [], []
    for i in line:
        Coordinate = CityCoordinates[i]
        x.append(Coordinate[0])
        y.append(Coordinate[1])
    x.append(x[0])
    y.append(y[0])
    plt.plot(x, y, 'r-', color='#4169E1', alpha=0.8, linewidth=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def crossover(bird, pLine, gLine, w, c1, c2,qwe):
    '''
    采用顺序交叉方式；交叉的parent1为粒子本身，分别以w/(w+c1+c2),c1/(w+c1+c2),c2/(w+c1+c2)
    的概率接受粒子本身逆序、当前最优解、全局最优解作为parent2,只选择其中一个作为parent2；
    输入：bird-粒子,pLine-当前最优解,gLine-全局最优解,w-惯性因子,c1-自我认知因子,c2-社会认知因子；
    输出：交叉后的粒子-croBird；交叉算子的作用是为了防止陷入局部最优
    '''
    croBird = [None] * len(bird)  # 初始化
    parent1 = bird  # 选择parent1
    # 选择parent2（轮盘赌操作）
    randNum = random.uniform(0, sum([w, c1, c2]))
    if randNum <= w:
        parent2 = [bird[i] for i in range(len(bird) - 1, -1, -1)]  # bird的逆序  因为正序进行交叉是两个相同的染色体，相当于没有交叉
    elif randNum <= w + c1:
        parent2 = pLine
    else:
        parent2 = gLine
    #if qwe<=3:
        #print(f'轮盘操作后的parent{parent2}')
    # parent1-> croBird遗传交叉算子
    start_pos = random.randint(0, len(parent1) - 1)  #随机生成起点
    end_pos = random.randint(0, len(parent1) - 1)   #随机生成终点
    if start_pos > end_pos: start_pos, end_pos = end_pos, start_pos    #将小的放在前面
    croBird[start_pos:end_pos + 1] = parent1[start_pos:end_pos + 1].copy()   #将区间内的parent1拷贝过去
    #if qwe<=3:
        #print(f'第一次操作后的croBird{croBird}')
    # parent2 -> croBird
    list1 = list(range(0, start_pos))   #生成需要填充的下标
    list2 = list(range(end_pos + 1, len(parent2)))
    list_index = list1 + list2  # croBird从后往前填充
    j = -1
    for i in list_index:
        for j in range(j + 1, len(parent2) + 1):
            if parent2[j] not in croBird:   #当croBird里有parent2[j]点重复，则不填充
                croBird[i] = parent2[j]     #将没有的点全部填充就去
                break
    #if qwe<=3:
        #print(f'第一次操作后的croBird{croBird}')
    qwe+=1
    return croBird,qwe


if __name__ == '__main__':
    # 参数
    CityNum = 200  # 城市数量
    MinCoordinate = 0  # 二维坐标最小值
    MaxCoordinate = 101  # 二维坐标最大值
    iterMax = 200  # 迭代次数
    iterI = 1  # 当前迭代次数

    # PSO参数
    birdNum = 50  # 粒子数量
    w = 0.2  # 惯性因子
    c1 = 0.4  # 自我认知因子
    c2 = 0.4  # 社会认知因子
    pBest, pLine = 0, []  # 当前最优值、当前最优解，（自我认知部分）
    gBest, gLine = 0, []  # 全局最优值、全局最优解，（社会认知部分）
    tracemalloc.start()
    # 随机生成城市数据,城市序号为0,1,2,3...
    #CityCoordinates = [(random.randint(MinCoordinate,MaxCoordinate),random.randint(MinCoordinate,MaxCoordinate)) for i in range(CityNum)]
    #print(CityCoordinates)
    begin_time = time()
    CityCoordinates = [(88, 16), (42, 76), (5, 76), (69, 13), (73, 56), (100, 100), (22, 92), (48, 74), (73, 46),(39, 1), (51, 75), (92, 2), (101, 44), (55, 26), (71, 27), (42, 81), (51, 91), (89, 54),(33, 18), (40, 78)]
    #CityCoordinates = [(1, 4), (2, 5), (3, 3), (4, 5), (5, 1), (6, 4)]
    #CityCoordinates = [[565.0,575.0],[25.0,185.0],[345.0,750.0],[945.0,685.0],[845.0,655.0],[880.0,660.0],[25.0,230.0],[525.0,1000.0],[580.0,1175.0],[650.0,1130.0],[1605.0,620.0],[1220.0,580.0],[1465.0,200.0],[1530.0,  5.0],[845.0,680.0],[725.0,370.0],[145.0,665.0],[415.0,635.0],[510.0,875.0],[560.0,365.0],[300.0,465.0],[520.0,585.0],[480.0,415.0],[835.0,625.0],[975.0,580.0],[1215.0,245.0],[1320.0,315.0],[1250.0,400.0],[660.0,180.0],[410.0,250.0],[420.0,555.0],[575.0,665.0],[1150.0,1160.0],[700.0,580.0],[685.0,595.0],[685.0,610.0],[770.0,610.0],[795.0,645.0],[720.0,635.0],[760.0,650.0],[475.0,960.0],[95.0,260.0],[875.0,920.0],[700.0,500.0],[555.0,815.0],[830.0,485.0],[1170.0, 65.0],[830.0,610.0],[605.0,625.0],[595.0,360.0],[1340.0,725.0],[1740.0,245.0]]
    #CityCoordinates = [(1, 5), (3, 4), (4, 0), (5, 3), (5, 6), (8, 5), (6, 1), (7, 3)]
    dis_matrix = calDistance(CityCoordinates)  # 计算城市间距离,生成矩阵
    #print(f'生成城市与城市之间的距离矩阵{dis_matrix}')
    birdPop = [random.sample(range(len(CityCoordinates)), len(CityCoordinates)) for i in range(birdNum)]  # 初始化种群，随机生成(生成点的序号)
    #print(f'随机生成商人对于城市的行走路线{birdPop}')
    fits = [calFitness(birdPop[i], dis_matrix) for i in range(birdNum)]  # 计算种群适应度,保存每种随机情况的总路程
    #print(f'记录每种路线的总路程长{fits}')
    gBest = pBest = min(fits)  # 全局最优值、当前最优值(总路程值)
    gLine = pLine = birdPop[fits.index(min(fits))]  # 全局最优解、当前最优解(走法)
    qwe=0
    while iterI <= iterMax:  # 迭代开始
        for i in range(len(birdPop)):
            birdPop[i],qwe = crossover(birdPop[i], pLine, gLine, w, c1, c2,qwe)   #遗传轮盘交叉
            fits[i] = calFitness(birdPop[i], dis_matrix)   #计算新种群适应度

        pBest, pLine = min(fits), birdPop[fits.index(min(fits))]    #更新当前最优值、最优解
        if min(fits) <= gBest:
            gBest, gLine = min(fits), birdPop[fits.index(min(fits))]   #更新全局最优值、最优解

        print(iterI, gBest)  # 打印当前代数和最佳适应度值
        iterI += 1  # 迭代计数加一

    #print(f'最优路径{gLine}')  # 路径顺序
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()
    end_time = time()
    run_time = end_time - begin_time
    print(run_time)
    draw_path(gLine, CityCoordinates)  # 画路径图
