#!/usr/bin/env python3
# *-* coding:utf8 *-*

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from time import *
import tracemalloc

from CONFIG.config import *
from UTIL.logger import Logger
from COMPONENT.application import Application
from service.applicationservice import ApplicationService
from system.computingsystem import ComputingSystem
from UTIL.genericutils import print_list
from SCHEDULER.task.heftscheduler import HeftScheduler
from SCHEDULER.task.cpopscheduler import CpopScheduler
from SCHEDULER.task.geneticscheduler import GeneticScheduler
from SCHEDULER.task.Evolutionscheduler import EvolutionScheduler
from SCHEDULER.task.wolfscheduler import wolfScheduler
from SCHEDULER.task.ASDE_GWO import ASDE_GWOcheduler
from SCHEDULER.task.poshscheduler import POSHScheduler
from datetime import datetime


def main():
    # 处理器数
    processor_number = 5
    # 计算系统初始化
    ComputingSystem.init(processor_number)
    # 生成应用
    appA = Application("A")
    appB = Application("B")
    appC = Application("C")
    appD = Application("D")

    '''HEFT'''
    # 任务执行时间矩阵
    # computation_time_matrix = [
    #     [14.00, 16.00, 9.00], [13.00, 19.00, 18.00], [11.00, 13.00, 19.00],
    #     [13.00, 8.00, 17.00], [12.00, 13.00, 10.00], [13.00, 16.00, 9.00],
    #     [7.00, 15.00, 11.00], [5.00, 11.00, 14.00], [18.00, 12.00, 20.00],
    #     [21.00, 7.00, 16.00]
    # ]
    computation_time_matrix = [[24, 22, 23, 16, 27], [28, 21, 9, 7, 16], [16, 25, 26, 10, 26], [19, 6, 11, 26, 21], [25, 6, 18, 18, 27],
     [26, 13, 14, 16, 25], [12, 24, 28, 25, 24], [25, 6, 7, 18, 19], [27, 27, 19, 28, 23], [13, 16, 26, 27, 20],
     [12, 27, 5, 28, 11], [20, 14, 16, 6, 17], [19, 18, 11, 25, 10], [7, 6, 20, 29, 25], [25, 29, 14, 14, 14],
     [8, 27, 14, 16, 20], [7, 8, 19, 9, 16], [25, 25, 24, 6, 28], [26, 21, 24, 13, 18], [29, 27, 10, 16, 16],
     [28, 18, 22, 6, 14], [7, 12, 20, 27, 7], [6, 26, 20, 26, 6], [25, 27, 6, 14, 19], [15, 9, 17, 25, 17],
     [8, 21, 27, 13, 25], [27, 7, 13, 11, 14], [29, 7, 24, 23, 25], [26, 29, 17, 23, 26], [6, 28, 18, 5, 7],
     [23, 27, 20, 15, 26], [25, 16, 8, 19, 17]]
    # 任务执行成本矩阵
    # computation_cost_matrix = [
    #     [14.00, 9.00, 16.00], [19.00, 13.00, 18.00], [19.00, 13.00, 11.00],
    #     [13.00, 17.00, 8.00], [12.00, 10.00, 13.00], [13.00, 9.00, 16.00],
    #     [15.00, 7.00, 11.00], [14.00, 11.00, 5.00], [18.00, 20.00, 12.00],
    #     [7.00, 21.00, 16.00]
    # ]
    computation_cost_matrix =[[23, 9, 20, 26, 26], [16, 5, 22, 28, 29], [24, 26, 14, 5, 22], [8, 18, 23, 13, 11], [7, 18, 7, 19, 14],
                              [20, 11, 15, 11, 22], [20, 18, 16, 23, 26], [29, 7, 24, 25, 18], [14, 24, 18, 18, 15], [17, 26, 12, 8, 19],
                              [12, 20, 18, 19, 22], [7, 12, 19, 27, 16], [9, 20, 21, 13, 17], [6, 10, 16, 26, 21], [17, 8, 5, 12, 13],
                              [18, 21, 27, 25, 24], [18, 7, 20, 8, 10], [21, 22, 18, 12, 17], [15, 5, 9, 15, 10], [22, 5, 11, 19, 19],
                              [12, 6, 7, 18, 16], [20, 26, 13, 18, 26], [11, 10, 8, 29, 15], [26, 21, 24, 25, 21], [13, 16, 15, 7, 7],
                              [7, 17, 11, 24, 15], [7, 15, 27, 5, 10], [22, 7, 25, 8, 9], [16, 10, 15, 20, 8], [21, 9, 10, 24, 29],
                              [27, 8, 5, 16, 22], [14, 25, 18, 14, 11]]
    # 任务通信时间矩阵
    # communication_time_matrix = [
    #     [INF, 18.00, 12.00, 9.00, 11.00, 14.00, INF, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, 19.00, 16.00, INF],
    #     [INF, INF, INF, INF, INF, INF, 23.00, INF, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, 27.00, 23.00, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, 13.00, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, 15.00, INF, INF],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, 17.00],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, 11.00],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, 13.00],
    #     [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    # ]
    communication_time_matrix = [
        [INF, 0.00, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, 40.00, INF, INF, INF, INF],
        [INF, INF, 40.00, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, 0.00, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, 45.00, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, 0.00, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 45.00, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, 55.00, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, 55.00, INF, INF, INF, 36.00, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, 0.00, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, 0.00, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0.00, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0.00, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0.00, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 36.00, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 45.00, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 45.00, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, 45.00, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0.00, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 0.00, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 55.00, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, 55.00,
         0.00, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, 0.00, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         0.00, INF, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, 55.00, INF, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, 45.00, INF, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, 45.00, INF, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, 45.00, INF, INF, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, 0.00],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, 0.00, INF, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, 40.00, INF, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, 0.00, INF],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, 40.00],
        [INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,
         INF, INF, INF, INF, INF, INF, INF, INF, INF, INF]
    ]
    # 任务数
    # task_number = 10
    task_number = 32
    # 初始化应用
    ApplicationService.init_application(appA, task_number, computation_time_matrix, computation_cost_matrix, communication_time_matrix)
    ApplicationService.init_application(appB, task_number, computation_time_matrix, computation_cost_matrix, communication_time_matrix)
    ApplicationService.init_application(appC, task_number, computation_time_matrix, computation_cost_matrix, communication_time_matrix)
    ApplicationService.init_application(appD, task_number, computation_time_matrix, computation_cost_matrix, communication_time_matrix)

    begin_time = time()
    tracemalloc.start()
    for i in range(10):
        # start_time = datetime.now()
        # print("-" * 100)
        # cpop = CpopScheduler("CPOP")
        # cpop_tradeoff = cpop.schedule(appA)
        # heft = HeftScheduler("HEFT")
        # heft_tradeoff = heft.schedule(appC)  # 调度器执行调度
        genetic = GeneticScheduler("Genetic")  #遗传调度
        ga_makespan, ga_cost = genetic.schedule(appC)
        # evolution=EvolutionScheduler("Evolution")  #差分进化调度
        # de_makespan, de_cost = evolution.schedule(appC)
        # wolf = wolfScheduler("wolf")  # 灰狼调度
        # GWO_makespan, GWO_cost = wolf.schedule(appC)
        # ASDE_GWO = ASDE_GWOcheduler("ASDE_GWO")  # 退火差分灰狼调度
        # ASDE_GWO_makespan, ASDE_GWO_cost = ASDE_GWO.schedule(appC)
        # posh = POSHScheduler("POSH")
        # posh_makespan, posh_cost = posh.schedule(appD)
        # end_time = datetime.now()
        # print("-" * 100)
        # print("<br/>%s ---- %s<br/>" % (start_time.strftime('%Y-%m-%d %H:%M:%S %f'), end_time.strftime('%Y-%m-%d %H:%M:%S %f')))
        # print("%s seconds" % (end_time - start_time).seconds)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()
    end_time = time()
    run_time = end_time - begin_time
    print(run_time)

    processor_list = ['8', '16', '32', '64', '128', '256', '512']
    slr_list_ga = [6.32, 6.02, 5.21, 4.46, 3.36, 2.66, 2.03]
    slr_list_posh = [6.54, 6.34, 5.73, 5.11, 4.30, 3.27, 2.64]
    mcr_list_ga = [1.56, 2.37, 3.46, 4.08, 4.82, 5.38, 5.61]
    mcr_list_posh = [1.66, 2.98, 4.17, 4.84, 5.91, 6.20, 6.66]

    '''
        颜色  color:修改颜色，可以简写成c
        样式  linestyle='--' 修改线条的样式 可以简写成 ls
        标注  marker : 标注
        线宽  linewidth: 设置线宽 可以简写成 lw   （lw=2）

    '''
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内, 必须放置在 title、xlabel、ylabel 之前才起作用
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内, 必须放置在 title、xlabel、ylabel 之前才起作用
    plt.xlabel("Alpha")
    plt.ylabel("TradeOff")

    plt.plot(processor_list, mcr_list_ga, c='#000000', linestyle='--', marker='o')
    plt.plot(processor_list, mcr_list_posh, c='#000000', linestyle='-.', marker='>')
    plt.legend(['NEGA', 'POSH'])
    plt.savefig("./result/GABUDGET_exp_result_mcr_processor.svg", dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
