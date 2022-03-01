#!/usr/bin/env python3
# *-* coding:utf8 *-*

"""
类别: 调度器
名称: CPOP调度器
作者: strong
邮件: jqjiang@hnist.edu.cn
日期: 2020年3月24日
说明: 来源Paper: Performance-effective and low-complexity task scheduling for heterogeneous computing
"""

from SCHEDULER.scheduler import Scheduler
from system.computingsystem import ComputingSystem
from UTIL.schedulerutils import SchedulerUtils
from COMPONENT.runningspan import RunningSpan
from COMPONENT.assignment import Assignment
from UTIL.genericutils import *
from CONFIG.config import *


class CpopScheduler(Scheduler):

    def schedule(self, app):

        ComputingSystem.reset(app)

        processors = ComputingSystem.processors

        tasks = app.prioritized_tasks  # 排序任务集
        tasks.sort(key=lambda tsk: tsk.rank_sum_value, reverse=True)    # 依据排序和值对任务进行降序排列

        critical_tasks = app.critical_tasks

        processor = None
        key_processor = SchedulerUtils.get_the_mini_computation_time_processor(critical_tasks, processors)  # 求得使关键任务集执行时间和值最小的处理器 —— 关键处理器

        i = 0

        while True:
            task = tasks[i]

            earliest_start_time = 0.0
            earliest_finish_time = float("inf")

            if SchedulerUtils.is_ready(task):   # 当当前遍历任务就绪时(即它所有的直接前驱任务均已被调度)
                if task.is_critical:    # 如果当前遍历任务为关键任务, 则:
                    processor = key_processor   # 将其安排在关键处理器上运行
                    earliest_start_time = SchedulerUtils.calculate_earliest_start_time(task, key_processor)
                    earliest_finish_time = SchedulerUtils.calculate_earliest_finish_time(task, key_processor)
                else:
                    for p in processors:
                        earliest_start_time_of_this_processor = SchedulerUtils.calculate_earliest_start_time(task, p)
                        earliest_finish_time_of_this_processor = SchedulerUtils.calculate_earliest_finish_time(task, p)
                        if earliest_finish_time > earliest_finish_time_of_this_processor:
                            processor = p
                            earliest_start_time = earliest_start_time_of_this_processor
                            earliest_finish_time = earliest_finish_time_of_this_processor                            

                i = 0
            else:
                i += 1
                continue

            running_span = RunningSpan(earliest_start_time, earliest_finish_time)

            assignment = Assignment(processor, running_span)

            task.assignment = assignment

            task.is_assigned = True

            processor.resident_tasks.append(task)
            processor.resident_tasks.sort(key=lambda tsk: tsk.assignment.running_span.start_time)

            self.task_scheduling_list.list[task] = assignment

            tasks.remove(task)

            if not tasks:
                break

        makespan = calculate_makespan(self.task_scheduling_list)
        cost = calculate_cost(self.task_scheduling_list)
        self.task_scheduling_list.makespan = makespan
        self.task_scheduling_list.cost = cost

        print("The scheduler = %s, list_name = %s, makespan = %.2f, cost = %.2f, tradeoff = %.2f" % (self.scheduler_name,
                                                                                    self.task_scheduling_list.list_name,
                                                                                    makespan, cost,
                                                                                    makespan*ALPHA + cost*BETA))

        # if SHOW_ORIGINAL_SCHEDULING_LIST:  # 如果打印原始调度列表, 则:
        #     print_scheduling_list(self.task_scheduling_list)  # 打印原始调度列表

        return makespan * ALPHA + cost * BETA
