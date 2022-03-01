#!/usr/bin/env python3
# *-* coding:utf8 *-*

"""
类别: 基本组件
名称: 处理器类
作者: strong
邮件: jqjiang@hnist.edu.cn
日期: 2020年3月20日
说明: 驻留任务集是被分配到处理器上的所有任务的集合
"""


class Processor(object):

    def __init__(self, name=None):
        self.name = name  # 处理器名称
        self.resident_tasks = []  # 处理器驻留任务集
        self.is_critical_path = False  # 是否为关键路径处理器

    def __str__(self):
        return "Processor [name = %s]" % self.name
