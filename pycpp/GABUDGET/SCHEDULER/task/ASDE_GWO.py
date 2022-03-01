#!/usr/bin/env python3
# *-* coding:utf8 *-*

"""
类别: 调度器
名称: 改进灰狼算法调度器 -- 针对任务的调度
作者: 孙质方
邮件: zf_sun@hnist.edu.cn
日期: 2021年12月19日
说明:
"""

import sys
import os
from SCHEDULER.scheduler import Scheduler
from system.computingsystem import ComputingSystem
from UTIL.schedulerutils import SchedulerUtils
from COMPONENT.runningspan import RunningSpan
from COMPONENT.assignment import Assignment
from COMPONENT.sequence import Sequence
from COMPONENT.schedulinglist import SchedulingList
from UTIL.genericutils import *
from UTIL.logger import Logger
from CONFIG.config import *

from itertools import permutations, product
from datetime import datetime
from copy import *
import random
import numpy
import math
from collections import Counter

class ASDE_GWOcheduler(Scheduler):

    # sys.stdout = Logger('./result/result_%d.html' % (random.randint(1000, 9999)))
    sys.stdout = Logger('D:/pycpp/GABUDGET/result/result_task.html')

    #def schedule(self, app,heft_list):
    def schedule(self, app):
        ComputingSystem.reset(app)  # 重置计算系统. !!!VERY IMPORTANT!!!
        pop_size = 10000
        #F = 0.8  # 缩放因子
        T = 1
        T0 = 1
        K = 0.97
        EPS = 0.00000001
        Gamma = math.ceil(math.log(EPS / T0, K))#总迭代次数
        #population = self.init_population(app, pop_size,heft_list)#1000*len(tasks)
        lenth = len(app.tasks)#任务个数
        population = self.init_population(app, pop_size,lenth)  # pop_size*len(tasks)
        best_population=population[0]
        Alpha_pos = population[0]
        Beta_pos = population[1]
        Delta_pos = population[2]
        l = 0
        best_ans=[]
        while T > EPS:
            if T <= 2 * EPS:
                target_population=self.learn(population,lenth,pop_size)
                self.reset_tasks(app.tasks)
                temp_task_list=[]
                temp_task_list.append(app.entry_task)
                node_size=[]
                for ind in range(len(target_population)//2):
                    ppp=len(temp_task_list)
                    node_size.append(ppp)
                    if ppp-1<target_population[ind]:
                        target_population[ind]=ppp-1
                    # print(f'{target_population[ind]}   {ppp}    {len(temp_task_list)}')
                    task1=temp_task_list[target_population[ind]]
                    task1.is_decoded=True
                    temp_task_list.remove(task1)
                    for succ in task1.successors:
                        if self.is_ready(succ):
                            temp_task_list.append(succ)
            temp_population = []
            half_popopulation=population[:pop_size//2]
            re_lenth = math.ceil((pop_size ) * (Gamma - l) / Gamma)  # re_lenth从pop_size线性减小到0
            if re_lenth%2:
                re_lenth+=1
            a = 2 - l * ((2) / Gamma)  # a从2线性减少到0
            for i in range(0,pop_size-re_lenth):
                temp_chromosome=[]
                for j in range(0, lenth*2):
                    r1 = random.random()  # r1 is a random number in [0,1]主要生成一个0-1的随机浮点数。
                    r2 = random.random()  # r2 is a random number in [0,1]

                    A1 = 2 * a * r1 - a  # (-a.a)
                    C1 = 2 * r2  # (0.2)
                    # D_alpha表示候选狼与Alpha狼的距离
                    D_alpha = abs(C1 * Alpha_pos.chromosome[j] - population[i].chromosome[j])  # abs() 函数返回数字的绝对值。Alpha_pos[j]表示Alpha位置，Positions[i,j])候选灰狼所在位置
                    X1 = Alpha_pos.chromosome[j] - A1 * D_alpha  # X1表示根据alpha得出的下一代灰狼位置向量

                    r1 = random.random()
                    r2 = random.random()

                    A2 = 2 * a * r1 - a  #
                    C2 = 2 * r2

                    D_beta = abs(C2 * Beta_pos.chromosome[j] - population[i].chromosome[j])
                    X2 = Beta_pos.chromosome[j] - A2 * D_beta

                    r1 = random.random()
                    r2 = random.random()

                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2

                    D_delta = abs(C3 * Delta_pos.chromosome[j] - population[i].chromosome[j])
                    X3 = Delta_pos.chromosome[j] - A3 * D_delta

                    # temp=((2 / 3) - (2 / 3) * (l / Gamma)) * X1 + (1 / 3) * X2 + (2 / 3) * (l / Gamma) * X3
                    temp=(2 / 3) * (l / Gamma) * X1 + (1 / 3) * X2 + ((2 / 3) - (2 / 3) * (l / Gamma)) * X3                    #根据迭代次数动态更新灰狼权值
                    if temp < 0:
                        temp = 0
                    elif temp >= 1:
                        temp = 0.9999999999
                    temp_chromosome.append(temp)  # 候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。
                temp_population.append(temp_chromosome)
            offspring_population = []
            for i in range(0,pop_size//2,2):
                prev_chromosome = []
                next_chromosome = []
                prev_chromosome.extend(population[i].chromosome)
                next_chromosome.extend(population[i + 1].chromosome)
                crossover_point = random.randint(1, len(prev_chromosome))
                for j in range(crossover_point):
                    prev_chromosome[j], next_chromosome[j] = next_chromosome[j], prev_chromosome[j]
                if T<=2*EPS:
                    if random.random()<math.exp(-T*(0.1/EPS)):
                        pos1 = random.randint(0, len(prev_chromosome)//2 - 1)
                        pos2 = random.randint(0, len(next_chromosome)//2 - 1)
                        prev_chromosome[pos1] = random.uniform(target_population[pos1]/node_size[pos1],target_population[pos1]/node_size[pos1]+1/node_size[pos1])
                        prev_chromosome[pos1+lenth] = random.uniform(target_population[pos1+lenth] / 5 ,target_population[pos1+lenth] / 5+ 1/5)
                        next_chromosome[pos2] = random.uniform(target_population[pos2]/node_size[pos2],target_population[pos2]/node_size[pos2]+1/node_size[pos2])
                        next_chromosome[pos2 + lenth] = random.uniform(target_population[pos2+lenth] / 5 ,target_population[pos2+lenth] / 5+ 1/5)
                    else:
                        pos1 = random.randint(0, len(prev_chromosome) - 1)
                        pos2 = random.randint(0, len(next_chromosome) - 1)
                        prev_chromosome[pos1] = random.random()
                        next_chromosome[pos2] = random.random()
                else:
                    pos1 = random.randint(0, len(prev_chromosome) - 1)
                    pos2 = random.randint(0, len(next_chromosome) - 1)
                    prev_chromosome[pos1] = random.random()
                    next_chromosome[pos2] = random.random()
                offspring_population.append(prev_chromosome)
                offspring_population.append(next_chromosome)
            temp_population+=offspring_population
            population = self.create_population(app, temp_population, lenth)
            population+=half_popopulation
            # re_poppopulation=self.re_gene(app,population,pop_size,lenth)
            # population+=self.create_population(app,re_poppopulation,lenth)
            population.sort(key=lambda seq: seq.tradeoff)
            population = population[:pop_size]
            Alpha_pos = population[0]
            Beta_pos = population[1]
            Delta_pos = population[2]
            if Alpha_pos.tradeoff<best_population.tradeoff:
                best_population=Alpha_pos
            T *= K
            l += 1
            best_ans.append(best_population)


            # print("<br/>generation = %d, makespan = %.2f, cost = %.2f, time = %s" % (k, population[0].makespan, population[0].cost, datetime.now().strftime('%Y-%m-%d %H:%M:%S %f')))

        # print("-" * 100)
        # print("<br/>pop_size = %d<br/>" % pop_size)
        # elite_sequence = population[0]
        elite_sequence = best_population
        makespan = elite_sequence.makespan#调度序列完工时间
        cost = elite_sequence.cost        #调度序列总成本
        tradeoff = ALPHA * makespan + BETA * cost
        print(elite_sequence.index_list)
        for i in range(len(best_ans)):
            print(best_ans[i].tradeoff,end=' ')
        print()
        print("The scheduler = %s, makespan = %.2f, cost = %.2f, tradeoff = %.2f, slr = %.2f, mcr = %.2f" % (self.scheduler_name,
                                                                                    makespan, cost, tradeoff,
                                                                                    makespan / app.cp_min_time,
                                                                                    cost / app.cp_min_cost))

        return makespan, cost

    #def init_population(self, app, pop_size,heft_list):
    def init_population(self, app, pop_size,lenth):
        chromosomes = []
        tasks = app.tasks

        for i in range(0, pop_size):  #种群数量1000
            chromosome = []
            for j in range(0, 2*lenth):
                chromosome.append(random.random())
            chromosomes.append(chromosome)
        #gene_list=self.re_index()
        population = self.create_population(app, chromosomes,lenth)#1000*len(tasks)
        temp_population=self.re_gene(app,population,pop_size,lenth)
        population+=self.create_population(app,temp_population,lenth)
        population.sort(key=lambda seq: seq.tradeoff)
        return population[:pop_size]

    def create_population(self, app, chromosomes,lenth):
        i = 0
        population = []
        candidate_tasks = []
        processor_set = ComputingSystem.processors#处理器
        while len(chromosomes) > 0:
            self.reset_tasks(app.tasks)
            candidate_tasks.clear()
            candidate_tasks.append(app.entry_task)#添加入口任务
            chromosome = chromosomes.pop(0)#取出chromosomes种群中第一个个体

            tsk_sequence = []
            prossor_sequence = []
            index_task=[]
            index_prossor=[]
            tast_list=[]
            prossor_list=[]
            for j in range(0, lenth):#chromosome整除2
                gene = chromosome[j]
                candidate_tasks.sort(key=lambda tsk: tsk.id)
                size = len(candidate_tasks)#入度为0的任务个数
                scale = 1.0 / size
                #print('$$$$$$')
                #print(f'{size}      {gene}')
                tsk_index = self.get_index(gene, scale, size)#???
                index_task.append(tsk_index)
                task = candidate_tasks[tsk_index]
                tast_list.append(task.id)
                task.is_decoded = True
                tsk_sequence.append(task)
                candidate_tasks.remove(task)
                for successor in task.successors:
                    if self.is_ready(successor):
                        candidate_tasks.append(successor)

                prossor_gene = chromosome[j+lenth]
                processor_set.sort(key=lambda prossor: prossor.id)
                prossor_size = len(processor_set)
                prossor_scale = 1.0 / prossor_size
                #print(f'$$${prossor_size}      {prossor_gene}')
                prossor_index = self.get_index(prossor_gene, prossor_scale, prossor_size)
                index_prossor.append(prossor_index)
                processor = processor_set[prossor_index]
                prossor_list.append(processor.id)
                prossor_sequence.append(processor)
            index_population=index_task+index_prossor
            index_list=tast_list+prossor_list
            makespan, cost = self.calculate_response_time_and_cost(app, i, tsk_sequence, prossor_sequence)
            i = i + 1
            s = Sequence(chromosome, index_population, index_list, tsk_sequence, prossor_sequence, makespan, cost)
            population.append(s)

        return population

    def reset_tasks(self, tasks):
        for task in tasks:
            task.is_decoded = False

    def is_ready(self, task):
        for predecessor in task.predecessors:
            if not predecessor.is_decoded:
                return False
        return True

    def get_index(self, gene, scale, size):#gene：个体的基因，scale=1/size
        l = 0
        r = size
        mid = (l + r) // 2
        while l <= r:
            temp = mid * scale
            if temp < gene:
                l = mid + 1
            elif temp > gene:
                r = mid - 1
            else:
                return mid
            mid = (l + r) // 2
        return mid


    # def re_index(self,heft_list,scale):
    #     gene_list=[]
    #     for i in heft_list:
    #         k=i
    #         gene=k*scale
    #         gene_list.append(gene)
    #     return gene_list

    def calculate_response_time_and_cost(self, app, counter, task_sequence, processor_sequence):
        ComputingSystem.reset(app)  # 重置计算系统. !!!VERY IMPORTANT!!!

        scheduling_list = self.scheduling_lists.setdefault(counter)#判断字典scheduling_lists里是否有键counter，没有自动添加

        if not scheduling_list:
            scheduling_list = SchedulingList("Scheduling_List_%d" % counter)
            self.scheduling_lists[counter] = scheduling_list

        for i in range(0, len(task_sequence)):  # 遍历当前消息分组内的所有消息
            task = task_sequence[i]  # 取任务 task
            processor = processor_sequence[i]  # 取任务 task 对应的运行处理器 processor

            start_time = SchedulerUtils.calculate_earliest_start_time(task, processor)  # 当前遍历处理器上最早可用启动时间
            finish_time = start_time + task.processor__computation_time[processor]  # 当前遍历处理器上最早可用结束时间

            running_span = RunningSpan(start_time, finish_time)  # 上述 for 循环结束后, 最合适的处理器已被找出, 此时可以记录下任务的运行时间段
            assignment = Assignment(processor, running_span)  # 同时记录下任务的运行时环境

            task.assignment = assignment  # 设置任务的运行时环境

            task.is_assigned = True  # 标记任务已被分配

            processor.resident_tasks.append(task)  # 将任务添加至处理器的驻留任务集中
            processor.resident_tasks.sort(key=lambda tsk: tsk.assignment.running_span.start_time)  # 对处理器的驻留任务进行排序, 依据任务启动时间升序排列

            self.scheduling_lists[counter].list[task] = assignment  # 将任务与对应运行时环境置于原始调度列表

        makespan = calculate_makespan(self.scheduling_lists[counter])
        cost = calculate_cost(self.scheduling_lists[counter])
        self.scheduling_lists[counter].makespan = makespan  # 计算原始调度列表的完工时间
        self.scheduling_lists[counter].cost = cost

        # print("The scheduler = %s, list_name = %s, makespan = %.2f" % (self.scheduler_name,
        #                                                                self.scheduling_lists[counter].list_name,
        #                                                                self.scheduling_lists[counter].makespan))

        # if SHOW_ORIGINAL_SCHEDULING_LIST:  # 如果打印原始调度列表, 则:
        #     print_scheduling_list(self.scheduling_lists[counter])  # 打印原始调度列表

        # if makespan < 100.0:
        #     for task in self.scheduling_lists[counter].list.keys():
        #         info = "%s\t%s\t%s" % (task.name, task.assignment.assigned_processor.name, task.assignment.running_span)
        #         print(info)
        #     print("-" * 100)
        #     print("The scheduler = %s, list_name = %s, makespan = %.2f<br/>" % (self.scheduler_name, self.scheduling_lists[counter].list_name, self.scheduling_lists[counter].makespan))
        #     print("#" * 100)
        # else:
        #     print("The scheduler = %s, list_name = %s, makespan = %.2f<br/>" % (self.scheduler_name, self.scheduling_lists[counter].list_name, self.scheduling_lists[counter].makespan))

        self.scheduling_lists.clear()
        return makespan, cost

    #生成对称基因
    def re_gene(self,app,Positions,pop_size,lenth):
        init_Positions = numpy.zeros((pop_size, 2*lenth))
        for i in range(2*lenth):
            for j in range(0,pop_size):
                if Positions[j].chromosome[i]>=(0 + 1)/2:
                    temp=(0 + 1)/2 - (Positions[j].chromosome[i]-(0 + 1)/2)
                    if temp<0:
                        temp=abs(temp)
                    if temp>1:
                        while temp > 1:
                            temp = 0 + temp - 1
                    if temp==1:
                        temp = 0.9999999999
                    init_Positions[j][i] = temp
                else:
                    temp = (0 + 1) / 2 + ((0 + 1) / 2-Positions[j].chromosome[i] )
                    if temp < 0:
                        temp = abs(temp)
                    if temp > 1:
                        while temp > 1:
                            temp = 0 + temp - 1
                    if temp == 1:
                        temp = 0.9999999999
                    init_Positions[j][i] = temp
        return list(init_Positions)
    #交叉
    def DE_crossover(self, population,v_list,CR):
        u_list = []
        vv_list = []
        amd=random.randint(0, len(v_list) - 1)
        for j in range(0, len(v_list)):
            if (random.random() <= CR) | (j == amd):
                # (j == random.randint(0, len_x - 1)是为了使变异中间体至少有一个基因遗传给下一代
                vv_list.append(v_list[j])
            else:
                vv_list.append(population[j])
        u_list.append(vv_list)
        return u_list

    def GA_crossover(self, population):
        offspring_population = []
        for i in range(0, len(population)-1, 2):
            prev_chromosome = []
            next_chromosome = []
            prev_chromosome.extend(population[i].chromosome)
            next_chromosome.extend(population[i+1].chromosome)
            crossover_point = random.randint(1, len(prev_chromosome))
            for j in range(crossover_point):
                prev_chromosome[j], next_chromosome[j] = next_chromosome[j], prev_chromosome[j]
            offspring_population.append(prev_chromosome)
            offspring_population.append(next_chromosome)

        return offspring_population

    def learn(self,popopulation,lenth,pop_size):
        temp_list=[]
        target_population=[]
        for i in range(2*lenth):
            for j in range(pop_size//2):
                temp_list.append(popopulation[j].index_population[i])
            collection_words = Counter(temp_list)
            most_counterNum = collection_words.most_common(1)
            target_population.append(most_counterNum[0][0])
        return target_population

# 列表相减
def substract(a_list, b_list,lb,ub):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        temp=a_list[i] - b_list[i]
        # if temp < lb:
        #     temp = lb
        # elif temp > ub:
        #     temp = ub
        # if temp < lb:
        #     temp = abs(temp)
        # if temp > ub:
        #     while temp>ub:
        #         temp = 0 + temp - 1
        # if temp == ub:
        #     temp = 0.9999999999
        new_list.append(temp)
    return new_list


# 列表相加
def add(a_list, b_list,lb,ub):
    a = len(a_list)
    new_list = []
    for i in range(0, a):
        temp = a_list[i] + b_list[i]
        # if temp < lb:
        #     temp = lb
        # elif temp > ub:
        #     temp = ub
        if temp < lb:
            temp = abs(temp)
        if temp > ub:
            while temp > ub:
                temp = 0 + temp - 1
        if temp == ub:
            temp = 0.9999999999
        new_list.append(temp)
    return new_list


# 列表的数乘
def multiply(a, b_list,lb,ub):
    b = len(b_list)
    new_list = []
    for i in range(0, b):
        temp = a * b_list[i]
        # if temp < lb:
        #     temp = lb
        # elif temp > ub:
        #     temp = ub
        # if temp < lb:
        #     temp = abs(temp)
        # if temp > ub:
        #     while temp > ub:
        #         temp = 0 + temp - 1
        # if temp == ub:
        #     temp = 0.9999999999
        new_list.append(temp)
    return new_list