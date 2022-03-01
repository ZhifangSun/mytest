# 模拟退火算法 程序：多变量连续函数优化
# Program: SimulatedAnnealing_v1.py
# Purpose: Simulated annealing algorithm for function optimization
# v1.0:
#   (1) 基本算法：单变量连续函数优化问题
#   (2) 文件输出优化结果和中间过程数据
#   (3) 设置指标参数计数器
#   (4) 图形输出坏解接受概率
# Copyright 2021 YouCans, XUPT
# Crated：2021-04-30

#  -*- coding: utf-8 -*-
import math  # 导入模块
import random  # 导入模块
import pandas as pd  # 导入模块 YouCans, XUPT
import numpy as np  # 导入模块 numpy, 并简写成 np
import matplotlib.pyplot as plt  # 导入模块 matplotlib.pyplot, 并简写成 plt
from datetime import datetime


# 子程序：定义优化问题的目标函数
def cal_Energy(X, nVar):
    # 测试函数 1： Schwefel 测试函数
    # -500 <= Xi <= 500
    # 全局极值：(420.9687,420.9687,...),f(x)=0.0
    cc = ss = 0
    for i in range(len(X)):
        ss += X[i] ** 2
        cc += math.cos(2 * math.pi * X[i])
    s = -20 * math.exp(-0.2 * math.sqrt(ss / len(X))) - math.exp(cc / len(X)) + 20 + math.exp(1);
    return s


# 子程序：模拟退火算法的参数设置
def ParameterSetting():
    cName = "funcOpt"  # 定义问题名称
    nVar = 30  # 给定自变量数量，y=f(x1,..xn)
    xMin = -32  # 给定搜索空间的下限，x1_min,..xn_min
    xMax = 32  # 给定搜索空间的上限，x1_max,..xn_max

    tInitial = 1000000.0  # 设定初始退火温度(initial temperature)
    tFinal = 1  # 设定终止退火温度(stop temperature)
    alfa = 0.98  # 设定降温参数，T(k)=alfa*T(k-1)
    meanMarkov = 100  # Markov链长度，也即内循环运行次数
    scale = 0.5  # 定义搜索步长，可以设为固定值或逐渐缩小
    return cName, nVar, xMin, xMax, tInitial, tFinal, alfa, meanMarkov, scale


# 模拟退火算法
def OptimizationSSA(nVar, xMin, xMax, tInitial, tFinal, alfa, meanMarkov, scale):
    # ====== 初始化随机数发生器 ======
    randseed = random.randint(1, 100)
    random.seed(randseed)  # 随机数发生器设置种子，也可以设为指定整数

    # ====== 随机产生优化问题的初始解 ======
    xInitial = np.zeros((nVar))  # 初始化，创建数组
    for v in range(nVar):
        # random.uniform(min,max) 在 [min,max] 范围内随机生成一个实数
        xInitial[v] = random.uniform(xMin, xMax)
    # 调用子函数 cal_Energy 计算当前解的目标函数值
    fxInitial = cal_Energy(xInitial, nVar)

    # ====== 模拟退火算法初始化 ======
    xNew = np.zeros((nVar))  # 初始化，创建数组
    xNow = np.zeros((nVar))  # 初始化，创建数组
    xBest = np.zeros((nVar))  # 初始化，创建数组
    xNow[:] = xInitial[:]  # 初始化当前解，将初始解置为当前解
    xBest[:] = xInitial[:]  # 初始化最优解，将当前解置为最优解
    fxNow = fxInitial  # 将初始解的目标函数置为当前值
    fxBest = fxInitial  # 将当前解的目标函数置为最优值
    print('x_Initial:{:.6f},{:.6f},\tf(x_Initial):{:.6f}'.format(xInitial[0], xInitial[1], fxInitial))

    recordIter = []  # 初始化，外循环次数
    recordFxNow = []  # 初始化，当前解的目标函数值
    recordFxBest = []  # 初始化，最佳解的目标函数值
    recordPBad = []  # 初始化，劣质解的接受概率
    kIter = 0  # 外循环迭代次数，温度状态数
    totalMar = 0  # 总计 Markov 链长度
    totalImprove = 0  # fxBest 改善次数
    nMarkov = meanMarkov  # 固定长度 Markov链

    # ====== 开始模拟退火优化 ======
    # 外循环，直到当前温度达到终止温度时结束
    tNow = tInitial  # 初始化当前温度(current temperature)
    l=0
    while tNow >= tFinal:  # 外循环，直到当前温度达到终止温度时结束
        # 在当前温度下，进行充分次数(nMarkov)的状态转移以达到热平衡
        kBetter = 0  # 获得优质解的次数
        kBadAccept = 0  # 接受劣质解的次数
        kBadRefuse = 0  # 拒绝劣质解的次数

        # ---内循环，循环次数为Markov链长度
        for k in range(nMarkov):  # 内循环，循环次数为Markov链长度
            totalMar += 1  # 总 Markov链长度计数器

            # ---产生新解
            # 产生新解：通过在当前解附近随机扰动而产生新解，新解必须在 [min,max] 范围内
            # 方案 1：只对 n元变量中的一个进行扰动，其它 n-1个变量保持不变
            xNew[:] = xNow[:]
            v = random.randint(0, nVar - 1)  # 产生 [0,nVar-1]之间的随机数
            for aa in range(len(xNew)):
                xNew[aa] = xNow[aa] + (xMax - xMin) * random.normalvariate(0, 1)
            # random.normalvariate(0, 1)：产生服从均值为0、标准差为 1 的正态分布随机实数
                xNew[aa] = max(min(xNew[aa], xMax), xMin)  # 保证新解在 [min,max] 范围内

            # ---计算目标函数和能量差
            # 调用子函数 cal_Energy 计算新解的目标函数值
            fxNew = cal_Energy(xNew, nVar)
            deltaE = fxNew - fxNow

            # ---按 Metropolis 准则接受新解
            # 接受判别：按照 Metropolis 准则决定是否接受新解
            if fxNew < fxNow:  # 更优解：如果新解的目标函数好于当前解，则接受新解
                accept = True
                kBetter += 1
            else:  # 容忍解：如果新解的目标函数比当前解差，则以一定概率接受新解
                pAccept = math.exp(-deltaE / tNow)  # 计算容忍解的状态迁移概率
                if pAccept > random.random():
                    accept = True  # 接受劣质解
                    kBadAccept += 1
                else:
                    accept = False  # 拒绝劣质解
                    kBadRefuse += 1

            # 保存新解
            if accept == True:  # 如果接受新解，则将新解保存为当前解
                xNow[:] = xNew[:]
                fxNow = fxNew
                if fxNew < fxBest:  # 如果新解的目标函数好于最优解，则将新解保存为最优解
                    fxBest = fxNew
                    xBest[:] = xNew[:]
                    totalImprove += 1
                    scale = scale * 0.99  # 可变搜索步长，逐步减小搜索范围，提高搜索精度

        # ---内循环结束后的数据整理
        # 完成当前温度的搜索，保存数据和输出
        pBadAccept = kBadAccept / (kBadAccept + kBadRefuse)  # 劣质解的接受概率
        recordIter.append(kIter)  # 当前外循环次数
        recordFxNow.append(round(fxNow, 4))  # 当前解的目标函数值
        recordFxBest.append(round(fxBest, 4))  # 最佳解的目标函数值
        recordPBad.append(round(pBadAccept, 4))  # 最佳解的目标函数值
        l+=0
        if kIter % 10 == 0:  # 模运算，商的余数
            print('i:{},t(i):{:.2f}, badAccept:{:.6f}, f(x)_best:{:.6f}'. \
                  format(kIter, tNow, pBadAccept, fxBest))

        # 缓慢降温至新的温度，降温曲线：T(k)=alfa*T(k-1)
        tNow = tNow * alfa
        kIter = kIter + 1
        # ====== 结束模拟退火过程 ======

    print('improve:{:d}'.format(totalImprove))
    return kIter, xBest, fxBest, fxNow, recordIter, recordFxNow, recordFxBest, recordPBad


# 结果校验与输出
def ResultOutput(cName, nVar, xBest, fxBest, kIter, recordFxNow, recordFxBest, recordPBad, recordIter):
    # ====== 优化结果校验与输出 ======
    fxCheck = cal_Energy(xBest, nVar)
    if abs(fxBest - fxCheck) > 1e-3:  # 检验目标函数
        print("Error 2: Wrong total millage!")
        return
    else:
        print("\nOptimization by simulated annealing algorithm:")
        for i in range(nVar):
            print('\tx[{}] = {:.6f}'.format(i, xBest[i]))
        print('\n\tf(x):{:.6f}'.format(fxBest))

    # ====== 优化结果写入数据文件 ======
    # nowTime = datetime.now().strftime('%m%d%H%M')  # '02151456'
    # fileName = "..\data\{}_{}.dat".format(cName, nowTime)  # 数据文件的地址和文件名
    # optRecord = {
    #     "iter": recordIter,
    #     "FxNow": recordFxNow,
    #     "FxBest": recordFxBest,
    #     "PBad": recordPBad}
    # df_Record = pd.DataFrame(optRecord)
    # df_Record.to_csv(fileName, index=False, encoding="utf_8_sig")
    # with open(fileName, 'a+', encoding="utf_8_sig") as fid:
    #     fid.write("\nOptimization by simulated annealing algorithm:")
    #     for i in range(nVar):
    #         fid.write('\n\tx[{}] = {:.6f}'.format(i, xBest[i]))
    #     fid.write('\n\tf(x):{:.6f}'.format(fxBest))
    # print("写入数据文件: %s 完成。" % fileName)

    # ====== 优化结果图形化输出 ======
    plt.figure(figsize=(6, 4), facecolor='#FFFFFF')  # 创建一个图形窗口
    plt.title('Optimization result: {}'.format(cName))  # 设置图形标题
    plt.xlim((0, kIter))  # 设置 x轴范围
    plt.xlabel('iter')  # 设置 x轴标签
    plt.ylabel('f(x)')  # 设置 y轴标签
    plt.plot(recordIter, recordFxNow, 'b-', label='FxNow')  # 绘制 FxNow 曲线
    plt.plot(recordIter, recordFxBest, 'r-', label='FxBest')  # 绘制 FxBest 曲线
    # plt.plot(recordIter,recordPBad,'r-',label='pBadAccept')  # 绘制 pBadAccept 曲线
    plt.legend()  # 显示图例
    plt.show()

    return


# 主程序
def main():
    # 参数设置，优化问题参数定义，模拟退火算法参数设置
    [cName, nVar, xMin, xMax, tInitial, tFinal, alfa, meanMarkov, scale] = ParameterSetting()
    # print([nVar, xMin, xMax, tInitial, tFinal, alfa, meanMarkov, scale])

    # 模拟退火算法
    [kIter, xBest, fxBest, fxNow, recordIter, recordFxNow, recordFxBest, recordPBad] \
        = OptimizationSSA(nVar, xMin, xMax, tInitial, tFinal, alfa, meanMarkov, scale)
    # print(kIter, fxNow, fxBest, pBadAccept)

    # 结果校验与输出
    ResultOutput(cName, nVar, xBest, fxBest, kIter, recordFxNow, recordFxBest, recordPBad, recordIter)


# === 关注 Youcans，分享更多原创系列 https://www.cnblogs.com/youcans/ ===
if __name__ == '__main__':
    main()

