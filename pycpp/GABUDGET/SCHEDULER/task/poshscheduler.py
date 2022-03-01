#!/usr/bin/env python3
# *-* coding:utf8 *-*

"""
类别: 调度器
名称: POSH算法调度器 -- 针对任务的调度
作者: strong
邮件: jqjiang@hnist.edu.cn
日期: 2021年4月7日
说明: Cost-efﬁcient task scheduling for executing large programs in the cloud
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


class POSHScheduler(Scheduler):

    sys.stdout = Logger('D:/pycpp/GABUDGET/result/result_task_posh.html')

    def schedule(self, app):
        ComputingSystem.reset(app)  # 重置计算系统. !!!VERY IMPORTANT!!!

        processors = ComputingSystem.processors  # 处理器集

        tasks = app.prioritized_tasks  # 排序任务集
        tasks.sort(key=lambda tsk: tsk.rank_up_value, reverse=True)

        for task in tasks:
            processor = None  # 全局处理器
            mini_tradeoff = INF
            for p in processors:
                temp_tradeoff = task.processor__tradeoff[p]
                if mini_tradeoff > temp_tradeoff:
                    mini_tradeoff = temp_tradeoff
                    processor = p

            earliest_start_time = SchedulerUtils.calculate_earliest_start_time(task, processor)
            earliest_finish_time = earliest_start_time + task.processor__computation_time[processor]

            running_span = RunningSpan(earliest_start_time, earliest_finish_time)  # 上述 for 循环结束后, 最合适的处理器已被找出, 此时可以记录下任务的运行时间段
            assignment = Assignment(processor, running_span)  # 同时记录下任务的运行时环境

            task.assignment = assignment  # 设置任务的运行时环境

            task.is_assigned = True  # 标记任务已被分配

            processor.resident_tasks.append(task)  # 将任务添加至处理器的驻留任务集中
            processor.resident_tasks.sort(key=lambda tsk: tsk.assignment.running_span.start_time)  # 对处理器的驻留任务进行排序, 依据任务启动时间升序排列

            self.task_scheduling_list.list[task] = assignment  # 将任务与对应运行时环境置于原始调度列表

        makespan = calculate_makespan(self.task_scheduling_list)
        cost = calculate_cost(self.task_scheduling_list)
        tradeoff = ALPHA * makespan + BETA * cost
        self.task_scheduling_list.makespan = makespan
        self.task_scheduling_list.cost = cost

        print("The scheduler = %s, makespan = %.2f, cost = %.2f, tradeoff = %.2f, slr = %.2f, mcr = %.2f" % (self.scheduler_name,
                                                                                                   makespan, cost, tradeoff,
                                                                                                   makespan / app.cp_min_time,
                                                                                                   cost / app.cp_min_cost))

        # if SHOW_ORIGINAL_SCHEDULING_LIST:  # 如果打印原始调度列表, 则:
        #     print_scheduling_list(self.task_scheduling_list)  # 打印原始调度列表

        return makespan, cost
