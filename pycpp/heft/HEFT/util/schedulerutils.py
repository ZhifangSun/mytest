#!/usr/bin/env python3
# *-* coding:utf8 *-*

"""
类别: 工具组件
名称: 调度器工具类
作者: strong
邮件: jqjiang@hnist.edu.cn
日期: 2020年3月23日
说明: 专门放置作用于调度器的工具函数的类
"""


class SchedulerUtils(object):

    # 计算实参任务在处理器上最早启动时间, 类方法
    @classmethod
    def calculate_earliest_start_time(cls, task, processor):

        predecessor_available_time = 0.0  # 变量, 用于记录前驱任务的最早可用时间
        processor_available_time = 0.0  # 变量, 用于记录处理器的最早可用时间

        resident_tasks = processor.resident_tasks  # 处理器的驻留任务集

        if resident_tasks:  # 如果驻留任务集不为空, 则:
            for t in resident_tasks:  # 遍历驻留任务集
                if t.application == task.application:  # 如果当前遍历驻留任务与实参任务属同一应用, 则:
                    temp_finish_time = t.assignment.running_span.finish_time
                    if processor_available_time < temp_finish_time:
                        processor_available_time = temp_finish_time  # 设置处理器的最早可用时间为最后一个驻留任务的结束时间

        if task.is_entry:  # 如果实参任务为入口任务, 则:
            earliest_start_time = processor_available_time  # 设置最早启动时间为处理器的最早可用时间
        else:  # 如果实参任务不为入口任务, 则:
            for predecessor in task.predecessors:  # 遍历实参任务的直接前驱任务集
                assigned_processor_of_predecessor = predecessor.assignment.assigned_processor  # 取前驱任务所分配运行处理器
                if processor == assigned_processor_of_predecessor:  # 如果实参处理器与前驱任务所分配运行处理器为同一处理器, 则:
                    communication_time = 0.0  # 置通信时间为 0
                else:  # 如果实参处理器与前驱任务所分配运行处理器不为同一处理器, 则:
                    communication_time = task.predecessor__communication_time[predecessor]  # 置通信时间为二者之间的通信时间

                finish_time_of_predecessor = predecessor.assignment.running_span.finish_time  # 取前驱任务的运行结束时间

                if (finish_time_of_predecessor + communication_time) > predecessor_available_time:  # 如果前驱任务运行结束时间加上通信时间大于此时前驱任务的最早可用时间, 则:
                    predecessor_available_time = finish_time_of_predecessor + communication_time  # 置前驱任务的最早可用时间为前驱任务运行结束时间与通信时间之和

            earliest_start_time = processor_available_time if processor_available_time > predecessor_available_time else predecessor_available_time  # 取前驱任务最早可用时间和处理器最早可用时间二者中最大值为实参任务最早启动时间

        return earliest_start_time  # 返回最早启动时间

    # 计算实参任务在处理器上最早结束时间, 类方法
    @classmethod
    def calculate_earliest_finish_time(cls, task, processor):
        earliest_start_time = cls.calculate_earliest_start_time(task, processor)
        computation_time = task.processor__computation_time[processor]
        earliest_finish_time = earliest_start_time + computation_time
        return earliest_finish_time

    @classmethod
    def get_the_mini_computation_time_processor(cls, tasks, processors):
        processor = None

        min_computation_time = float("inf")

        for p in processors:
            computation_time = 0.0
            for task in tasks:
                computation_time += task.processor__computation_time[p]

            if min_computation_time > computation_time:
                min_computation_time = computation_time
                processor = p

        return processor

    @classmethod
    def is_ready(cls, task):
        flag = True
        if task.predecessors:
            for predecessor in task.predecessors:
                flag = flag and predecessor.is_assigned

        return flag

    @classmethod
    def get_the_max_steps_to_the_entry(cls, predecessors):
        max_steps = 0

        if not predecessors:
            max_steps = 0
        else:
            for predecessor in predecessors:
                steps = cls.get_the_max_steps_to_the_entry(predecessor.predecessors) + 1
                if steps > max_steps:
                    max_steps = steps

        return max_steps

    @classmethod
    def get_the_max_steps_to_the_exit(cls, successors):
        max_steps = 0

        if not successors:
            max_steps = 0
        else:
            for successor in successors:
                steps = cls.get_the_max_steps_to_the_exit(successor.successors) + 1
                if steps > max_steps:
                    max_steps = steps

        return max_steps
