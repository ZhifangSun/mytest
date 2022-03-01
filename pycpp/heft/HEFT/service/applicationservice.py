#!/usr/bin/env python3
# *-* coding:utf8 *-*

"""
类别: 服务组件
名称: 应用服务类
作者: strong
邮件: jqjiang@hnist.edu.cn
日期: 2020年3月21日
说明: 对应用进行初始化等操作的服务工具类
"""

from component.application import Application
from component.task import Task
from component.group import Group
from system.computingsystem import ComputingSystem
from util.genericutils import print_list
from util.schedulerutils import SchedulerUtils
import numpy as np


class ApplicationService(object):
    # 初始化应用, 静态方法
    @staticmethod
    def init_application(app, task_number, processor_number, computation_time_matrix, communication_time_matrix):
        OCT = np.zeros((task_number, processor_number))
        ApplicationService.__init_task_list(app, task_number)                       # 初始化任务列表
        ApplicationService.__init_computation_time(app, computation_time_matrix)    # 初始化执行时间
        ApplicationService.__init_successor(app, communication_time_matrix)         # 初始化后继任务
        ApplicationService.__init_predecessor(app)                                  # 初始化前驱任务
        ApplicationService.__calculate_rank_up_value(app)                           # 计算任务向上排序值
        ApplicationService.__calculate_rank_OCT_value(app, OCT, processor_number)                     # 计算任务OCT排序值
        ApplicationService.__calculate_rank_down_value(app)                         # 计算任务向下排序值
        ApplicationService.__calculate_rank_sum_value(app)
        ApplicationService.__tag_entry_and_exit_task(app)                           # 标记入口和出口任务
        ApplicationService.__tag_critical_task(app,computation_time_matrix)
        # ApplicationService.__sort_tasks(app)                                        # 排序任务
        ApplicationService.__group_task_from_the_top(app)
        ApplicationService.__group_task_from_the_bottom(app)

    # 初始化任务列表, 静态私有方法
    @staticmethod
    def __init_task_list(app, task_number):
        for i in range(task_number):
            t = Task("%s-task-%d" % (app.name, (i + 1)))  # 给任务取名
            t.num=i
            t.application = app  # 设置任务所属应用
            app.tasks.append(t)  # 将任务添加至应用任务集
            app.prioritized_tasks.append(t)  # 同时将任务添加至应用排序任务集

    # 初始化任务执行时间, 静态私有方法
    @staticmethod
    def __init_computation_time(app, computation_time_matrix):

        if not computation_time_matrix:  # 如果执行时间矩阵为空, 则:
            return  # 直接返回, 什么也不做

        processors = ComputingSystem.processors  # 处理器集3

        for i in range(len(app.tasks)):  # 遍历应用任务集10
            total = 0
            task = app.tasks[i]  # 取当前遍历任务
            for j in range(len(processors)):  # 遍历处理器集
                processor = processors[j]  # 取当前遍历处理器
                task.processor__computation_time[processor] = computation_time_matrix[i][j]  # 设置任务的 处理器-执行时间值 为矩阵中对应值
                total += computation_time_matrix[i][j]  # 同时将该值累加至总值中
            task.average_computation_time = total / len(processors)  # 设置任务的平均执行时间

    # 初始化任务的直接后继任务, 静态私有方法
    @staticmethod
    def __init_successor(app, communication_time_matrix):

        if not communication_time_matrix:  # 如果通信时间矩阵为空, 则:
            return  # 直接返回, 什么也不做

        for i in range(len(app.tasks)):  # 遍历任务集10
            task = app.tasks[i]  # 取当前遍历任务
            communication_times = communication_time_matrix[i]  # 取当前遍历任务的通信时间行值
            for j in range(len(communication_times)):  # 遍历通信时间行值10
                if communication_times[j] > 0:  # 如果存在通信时间, 则:
                    successor = app.tasks[j]  # 根据 j 值获得一个直接后继任务
                    task.successor__communication_time[successor] = communication_times[j]  # 设置任务的 后继任务-通信时间值 为矩阵中对应值
                    task.successors.append(successor)  # 将后继任务置入任务的直接后继任务集

            task.out_degree = len(task.successors)  # 最后根据直接后继任务个数, 求得任务的出度

    # 初始化任务的直接前驱任务, 静态私有方法
    @staticmethod
    def __init_predecessor(app):
        for i in range(len(app.tasks)):  # 遍历任务集10
            task = app.tasks[i]  # 取当前遍历任务
            for successor in task.successors:  # 遍历当前任务的直接后继任务集
                communication_time = task.successor__communication_time[successor]  # 取与后继任务的通信时间
                successor.predecessor__communication_time[task] = communication_time  # 设置后继任务的 前驱任务-通信时间值
                successor.predecessors.append(task)  # 将当前遍历任务添加至后继任务的直接前驱任务集中

            task.in_degree = len(task.predecessors)  # 最后根据直接前驱任务个数，求得任务的入度

    # 计算任务的向上排序值, 静态私有化方法
    @staticmethod
    def __calculate_rank_up_value(app):
        for i in range(len(app.tasks) - 1, -1, -1):  # 倒序遍历任务集, 注意: 终点值是 -1 , 不是 0
            task = app.tasks[i]  # 取当前遍历任务
            temp_value = 0.0  # 临时变量, 用于保存临时向上排序值
            if task.successors:  # 如果当前任务存在直接后继任务集, 则:
                for successor in task.successors:  # 遍历后继任务集
                    successor_rank_up_value = successor.rank_up_value  # 取后继任务的向上排序值0.0
                    successor_communication_time = task.successor__communication_time[successor]  # 取与后继任务的通信时间
                    if (successor_rank_up_value + successor_communication_time) > temp_value:  # 如果后继任务的向上排序值与通信时间之和大于临时变量值, 则:
                        temp_value = successor_rank_up_value + successor_communication_time  # 置临时变量值为后继任务的向上排序值与通信时间之和,整个if语句的意思是取最大值

            task.rank_up_value = task.average_computation_time + temp_value  # 设置任务的向上排序值为平均执行时间与临时变量值之和

    # 计算任务的OCT排序值, 静态私有方法
    @staticmethod
    def __calculate_rank_OCT_value(app, OCT, processor_number):
        processors = ComputingSystem.processors
        for i in range(len(app.tasks) - 1, -1, -1):
            task = app.tasks[i]
            for j in range(processor_number):
                if task.successors:
                    for successor in task.successors:
                        ans=9999999999
                        for w in range(processor_number):
                            if w == j:
                                a=OCT[successor.num,w]+successor.processor__computation_time[processors[w]]
                                if a<ans:
                                    ans=a
                            else:
                                a = OCT[successor.num, w] + successor.processor__computation_time[processors[w]] + task.successor__communication_time[successor]
                                if a<ans:
                                    ans=a
                        if ans>OCT[i,j]:
                            OCT[i, j]=ans
        print(OCT)
    # 计算任务的向下排序值, 静态私有方法
    @staticmethod
    def __calculate_rank_down_value(app):
        for task in app.tasks:
            temp_value = 0.0
            if task.predecessors:    # 如果当前任务存在直接前驱任务集, 则:
                for predecessor in task.predecessors:
                    average_computation_time_of_predecessor = predecessor.average_computation_time   # 任务平均处理时间
                    rank_down_value_of_predecessor = predecessor.rank_down_value          # 取前驱任务的向下排序值0.0
                    communication_time_of_predecessor = task.predecessor__communication_time[predecessor]
                    if (rank_down_value_of_predecessor + average_computation_time_of_predecessor + communication_time_of_predecessor) > temp_value:
                        temp_value = rank_down_value_of_predecessor + average_computation_time_of_predecessor + communication_time_of_predecessor   # 向下排序值 = max(直接前驱任务的向下排序值 + 直接前驱任务的平均执行时间 + 直接前驱任务的通信时间值)

            task.rank_down_value = temp_value

    @staticmethod
    def __calculate_rank_sum_value(app):
        for task in app.tasks:
            task.rank_sum_value = task.rank_up_value + task.rank_down_value

    # 标记入口和出口任务, 静态私有方法
    @staticmethod
    def __tag_entry_and_exit_task(app):
        for task in app.tasks:  # 遍历任务集
            if not task.predecessors:  # 如果任务的直接前驱任务集为空, 则
                task.is_entry = True  # 置任务为入口任务
                app.entry_tasks.append(task)  # 同时将任务添加至应用的入口任务集
            if not task.successors:  # 如果任务的直接后继任务集为空, 则
                task.is_exit = True  # 置任务为出口任务
                app.exit_tasks.append(task)  # 同时将任务添加至应用的出口任务集

    # 标记关键人物, 静态私有方法
    @staticmethod
    def __tag_critical_task(app,computation_time_matrix):
        task = Task()
        rank_sum_value = 0.0

        for entry in app.entry_tasks:   # 遍历入口任务集
            if entry.rank_sum_value > rank_sum_value:   # 如果入口任务的排序和值大于变量值, 则:
                task = entry    # 置变量值为当前遍历入口任务
                rank_sum_value = entry.rank_sum_value   # 置变量值为当前遍历入口任务的向上排序和值

        task.is_critical = True
        app.critical_tasks.append(task) # 往应用关键任务集里添加任务
        '''
        以下 while 循环, 通过不断层层遍历任务的后继任务, 从当中找出关键任务, 直到出口任务为止中断循环
        '''
        print(app.tasks[8].rank_sum_value)
        print(app.tasks[9].rank_sum_value)
        while True:
            temp_task = Task()
            temp_rank_sum_value = task.rank_sum_value
            print(temp_rank_sum_value)
            print(task.num)
            for successor in task.successors:
                if successor.rank_sum_value >= temp_rank_sum_value:
                    print(successor.num)
                    temp_task = successor
                    temp_rank_sum_value = successor.rank_sum_value

            task = temp_task
            task.is_critical = True
            app.critical_tasks.append(task)

            if task.is_exit:
                break
            if not task.successors:
                break
        print("%%%%%%")
        EFT=[0]*len(ComputingSystem.processors)
        for i in app.critical_tasks:
            print(i.num)
            for j in range(len(ComputingSystem.processors)):
                EFT[j]+=computation_time_matrix[i.num][j]
        print(EFT)
        ComputingSystem.processors[EFT.index(min(EFT))].is_critical_path=True
    # 对任务排序, 静态私有方法
    @staticmethod
    def __sort_tasks(app):
        app.prioritized_tasks.sort(key=lambda task: task.rank_up_value, reverse=True)  # 针对应用排序任务集中的任务, 根据其向上排序值降序排列
        # sorted(app.prioritized_tasks, key=lambda task: task.rank_up_value)           # 另外一种排序方式

    @staticmethod
    def __group_task_from_the_top(app):
        groups = app.task_groups_from_the_top
        tasks = app.tasks
        for task in tasks:
            k = SchedulerUtils.get_the_max_steps_to_the_entry(task.predecessors)
            group = groups.setdefault(k)    #查询字典groups里是否含有key=k的键，有就返回k对应的value，没有就默认添加键k，值为None
            if not group:
                group = Group(k)
                groups[k] = group

            group.tasks.append(task)

    @staticmethod
    def __group_task_from_the_bottom(app):
        groups = app.task_groups_from_the_bottom
        tasks = app.tasks
        for task in tasks:
            k = SchedulerUtils.get_the_max_steps_to_the_exit(task.successors)
            group = groups.setdefault(k)
            if not group:
                group = Group(k)
                groups[k] = group

            group.tasks.append(task)
