#!/usr/bin/env python3
# *-* coding:utf8 *-*

"""
类别: 工具组件
名称: 通用工具
作者: strong
邮件: jqjiang@hnist.edu.cn
日期: 2020年3月20日
"""


# 打印列表函数
def print_list(object_list):
    for o in object_list:
        print(o)


# 计算调度序列完工时间函数
def calculate_makespan(scheduling_list):
    makespan = 0.0
    for task in scheduling_list.list.keys():
        if task.is_exit:
            finish_time = task.assignment.running_span.finish_time
            makespan = finish_time if finish_time > makespan else makespan
    return makespan


# 打印调度序列
def print_scheduling_list(scheduling_list):
    for task in scheduling_list.list.keys():
        info = "%s to the %s of %s" % (task, task.assignment.assigned_processor, task.assignment.running_span)
        print(info)
