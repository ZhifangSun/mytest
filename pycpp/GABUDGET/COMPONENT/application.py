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
        self.entry_task = None  # 应用入口任务
        self.exit_task = None  # 应用出口任务
        self.tasks = []  # 应用任务集
        self.authenticated_tasks = []
        self.prioritized_tasks = []  # 排序后任务集
        self.critical_tasks = []  # 应用关键任务集
        self.deadline = 0.0  # 应用截止时间
        self.budget = 0.0  # 应用可用成本
        self.task_groups_from_the_top = {}
        self.task_groups_from_the_bottom = {}

        self.all_messages = []  # 应用的消息集
        self.valid_messages = []    # 应用的有效消息集
        self.prioritized_messages = []  # 排序后消息集
        self.temp_messages = []     # 用于转换时存放半成品消息
        self.message_groups_from_the_top = {}

        self.sequences = []

        self.cp_min_time = 0.0  # 关键路径任务最小执行时间之和
        self.cp_min_cost = 0.0  # 关键路径任务最小执行成本之和

    def __str__(self):
        return "Application [name = %s]" % self.name
