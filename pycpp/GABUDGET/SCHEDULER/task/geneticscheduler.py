#!/usr/bin/env python3
# *-* coding:utf8 *-*

"""
类别: 调度器
名称: 遗传算法调度器 -- 针对任务的调度
作者: strong
邮件: jqjiang@hnist.edu.cn
日期: 2020年11月22日
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


class GeneticScheduler(Scheduler):

    # sys.stdout = Logger('./result/result_%d.html' % (random.randint(1000, 9999)))
    sys.stdout = Logger('D:/pycpp/GABUDGET/result/result_task.html')

    def schedule(self, app):
        ComputingSystem.reset(app)  # 重置计算系统. !!!VERY IMPORTANT!!!
        pop_size = 3000
        population = self.init_population(app, pop_size)
        k = 0
        best_ans=[]
        while k < 150:
            half_population = []
            half_population.extend(self.select(population))#每次选择种群里适应度值靠前的一半
            crossover_chromosomes = self.crossover(half_population)#前后交叉
            mutation_chromosomes = self.mutate(crossover_chromosomes)#个体变异
            population = population[:len(population)//2]
            population.extend(self.create_population(app, mutation_chromosomes))
            population.sort(key=lambda seq: seq.tradeoff)
            # print("<br/>generation = %d, makespan = %.2f, cost = %.2f, time = %s" % (k, population[0].makespan, population[0].cost, datetime.now().strftime('%Y-%m-%d %H:%M:%S %f')))
            k = k + 1
            best_ans.append(population[0])

        # print("-" * 100)
        # print("<br/>pop_size = %d<br/>" % pop_size)
        elite_sequence = population[0]
        makespan = elite_sequence.makespan#调度序列完工时间
        cost = elite_sequence.cost#调度序列总成本
        tradeoff = ALPHA * makespan + BETA * cost
        for i in range(len(best_ans)):
            print(best_ans[i].tradeoff,end=' ')
        print()
        print("The scheduler = %s, makespan = %.2f, cost = %.2f, tradeoff = %.2f, slr = %.2f, mcr = %.2f" % (self.scheduler_name,
                                                                                    makespan, cost, tradeoff,
                                                                                    makespan / app.cp_min_time,
                                                                                    cost / app.cp_min_cost))

        return makespan, cost

    def init_population(self, app, pop_size):
        chromosomes = []
        tasks = app.tasks

        for i in range(0, pop_size):  #种群数量1000
            chromosome = []
            for j in range(0, 2 * len(tasks)):#两倍的任务数量
                chromosome.append(random.random())
            chromosomes.append(chromosome)

        population = self.create_population(app, chromosomes)#1000*len(tasks)
        population.sort(key=lambda seq: seq.tradeoff)
        return population

    def create_population(self, app, chromosomes):
        i = 0
        population = []
        candidate_tasks = []
        processor_set = ComputingSystem.processors#处理器
        while len(chromosomes) > 0:    #2 * len(tasks)
            self.reset_tasks(app.tasks)
            candidate_tasks.clear()
            candidate_tasks.append(app.entry_task)#添加入口任务
            chromosome = chromosomes.pop(0)#取出chromosomes种群中第一个个体

            tsk_sequence = []
            for j in range(0, len(chromosome)//2):#chromosome整除2
                gene = chromosome[j]
                candidate_tasks.sort(key=lambda tsk: tsk.id)
                size = len(candidate_tasks)#入度为0的任务个数
                scale = 1.0 / size
                tsk_index = self.get_index(gene, scale, size)#???
                task = candidate_tasks[tsk_index]
                task.is_decoded = True
                tsk_sequence.append(task)
                candidate_tasks.remove(task)
                for successor in task.successors:
                    if self.is_ready(successor):
                        candidate_tasks.append(successor)

            prossor_sequence = []
            for j in range(len(chromosome)//2, len(chromosome)):
                prossor_gene = chromosome[j]
                processor_set.sort(key=lambda prossor: prossor.id)
                prossor_size = len(processor_set)
                prossor_scale = 1.0 / prossor_size
                prossor_index = self.get_index(prossor_gene, prossor_scale, prossor_size)
                processor = processor_set[prossor_index]
                prossor_sequence.append(processor)

            makespan, cost = self.calculate_response_time_and_cost(app, i, tsk_sequence, prossor_sequence)
            i = i + 1
            s = Sequence(chromosome, tsk_sequence, prossor_sequence, makespan, cost)
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

    def get_index(self, gene, scale, size):#gene个体的基因，scale=1/size
        l = 0
        r = size
        mid=(l + r) // 2
        while l <= r:
            temp = mid * scale
            if temp < gene:
                l = mid+1
            elif temp > gene:
                r = mid-1
            else:
                return mid
            mid = (l + r) // 2
        return mid
        # i = 1
        # while i <= size:
        #     if i*scale < gene:
        #         i = i + 1
        #     else:
        #         break
        # return i-1

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

    def select(self, population):
        half_population = population[:len(population)//2]
        return half_population

    def crossover(self, population):
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

    def mutate(self, population):
        for p in population:
            pos = random.randint(0, len(p)-1)
            p[pos] = random.random()
        return population
        pass
