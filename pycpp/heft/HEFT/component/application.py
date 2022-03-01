#!/usr/bin/env python3
# *-* coding:utf8 *-*

"""
类别: 基本组件
名称: 应用类
作者: strong
邮件: jqjiang@hnist.edu.cn
日期: 2020年3月21日
说明: 
"""


class Application(object):

    def __init__(self, name=None):
        self.name = name  # 应用名称
        self.criticality = 0  # 应用重要/紧急程度
        self.entry_tasks = []  # 应用入口任务集
        self.exit_tasks = []  # 应用出口任务集
        self.tasks = []  # 应用任务集
        self.prioritized_tasks = []  # 应用排序后任务集
        self.critical_tasks = []  # 应用关键任务集
        self.deadline = 0.0  # 应用截止时间
        self.budget = 0.0  # 应用可用成本
        self.task_groups_from_the_top = {}
        self.task_groups_from_the_bottom = {}

    def __str__(self):
        return "Application [name = %s]" % self.name
