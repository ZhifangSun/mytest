#!/usr/bin/env python3
# *-* coding:utf8 *-*

"""
类别: 基本组件
名称: 任务类
作者: strong
邮件: jqjiang@hnist.edu.cn
日期: 2020年3月20日
说明: 重要的组件类
"""


class Task(object):

    def __init__(self, name=None):
        self.num=0
        self.name = name  # 任务名称
        self.application = None  # 任务所属应用
        self.is_entry = False  # 是否为入口任务
        self.is_exit = False  # 是否为出口任务
        self.is_critical = False  # 是否为关键任务
        self.is_pseudo = False  # 是否为伪任务
        self.is_assigned = False  # 是否已被分配
        self.processor__computation_time = {}  # 处理器-执行时间值对
        self.processor__computation_cost = {}  # 处理器-执行成本值对
        self.processor__rank_up_value = {}  # 处理器-向上排序值对
        self.successor__communication_time = {}  # 直接后继任务-通信时间值对
        self.predecessor__communication_time = {}  # 直接前驱任务-通信时间值对
        self.successors = []  # 任务的直接后继任务集
        self.predecessors = []  # 任务的直接前驱任务集
        self.in_degree = 0  # 任务入度
        self.out_degree = 0  # 任务出度
        self.average_computation_time = 0.0  # 任务平均执行时间
        self.rank_up_value = 0.0  # 任务向上排序值
        self.rank_down_value = 0.0  # 任务向下排序值
        self.rank_sum_value = 0.0  # 任务向上向下排序值之和
        self.assignment = None  # 任务运行时环境

    def __str__(self):
        return "Task [name = %s]" % self.name
