#!/usr/bin/env python3
# *-* coding:utf8 *-*

"""
类别: 调度器组件
名称: 调度器基类
作者: strong
邮件: jqjiang@hnist.edu.cn
日期: 2020年3月21日
说明: 所有具体调度器的基类
"""

from component.schedulinglist import SchedulingList


class Scheduler(object):

    def __init__(self, scheduler_name):
        self.scheduler_name = scheduler_name  # 调度器名称
        self.original_scheduling_list = SchedulingList("OriginalSchedulingList")  # 原始调度列表
        self.optimized_scheduling_list = SchedulingList("OptimizedSchedulingList")  # 优化调度列表
