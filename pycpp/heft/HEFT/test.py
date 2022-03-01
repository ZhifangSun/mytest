#!/usr/bin/env python3
# *-* coding:utf8 *-*

from component.application import Application
from service.applicationservice import ApplicationService
from system.computingsystem import ComputingSystem
from util.genericutils import print_list
from SCHEDULER.static.heftscheduler import HeftScheduler
from SCHEDULER.static.cpopscheduler import CpopScheduler


def main():
    # 处理器数
    processor_number = 3
    # 计算系统初始化
    ComputingSystem.init(processor_number)
    # 生成应用
    app = Application("H")
    # 任务执行时间矩阵
    # computation_time_matrix = [[14.00, 16.00, 9.00],
    #                            [13.00, 19.00, 18.00],
    #                            [11.00, 13.00, 19.00],
    #                            [13.00, 8.00, 17.00],
    #                            [12.00, 13.00, 10.00],
    #                            [13.00, 16.00, 9.00],
    #                            [7.00, 15.00, 11.00],
    #                            [5.00, 11.00, 14.00],
    #                            [18.00, 12.00, 20.00],
    #                            [21.00, 7.00, 16.00]]
    computation_time_matrix = [[22.00, 21.00, 36.00],
                               [22.00, 18.00, 18.00],
                               [32.00, 27.00, 43.00],
                               [7.00, 10.00, 4.00],
                               [29.00, 27.00, 35.00],
                               [26.00, 17.00, 24.00],
                               [14.00, 25.00, 30.00],
                               [29.00, 23.00, 36.00],
                               [15.00, 21.00, 8.00],
                               [13.00, 16.00, 33.00]]
    # 任务通信时间矩阵
    # communication_time_matrix = [
    #     [0.00, 18.00, 12.00, 9.00, 11.00, 14.00, 0.00, 0.00, 0.00, 0.00],
    #     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 19.00, 16.00, 0.00],
    #     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 23.00, 0.00, 0.00, 0.00],
    #     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 27.00, 23.00, 0.00],
    #     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 13.00, 0.00],
    #     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 15.00, 0.00, 0.00],
    #     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 17.00],
    #     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.00],
    #     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 13.00],
    #     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    # ]
    communication_time_matrix = [
        [0.00, 17.00, 31.00, 29.00, 13.00, 7.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 3.00, 30.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 16.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.00, 7.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 57.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 5.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 9.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 42.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 7.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    ]
    # 任务数
    task_number = 10
    # 初始化应用
    ApplicationService.init_application(app, task_number, processor_number, computation_time_matrix, communication_time_matrix)

    for t in app.tasks:
        # print(t.processor__computation_time[ComputingSystem.processors[0]])
        # print(t.average_computation_time)
        # print("task = %s, rankU = %f, rankD = %f, rankSum = %f" % (t, t.rank_up_value, t.rank_down_value, t.rank_sum_value))
        pass

    for key, group in app.task_groups_from_the_top.items():
        print("group_id = %d" % key)
        for task in group.tasks:
            print("task = %s" % task, end=" ")
        print()

    # print_list(app.entry_tasks)
    # print_list(app.exit_tasks)
    # print_list(app.critical_tasks)

    # 调度器生成
    heft = HeftScheduler("HEFT")
    heft.schedule(app)  # 调度器执行调度

    # cpop=CpopScheduler("CPOP")
    # cpop.schedule(app)


if __name__ == '__main__':
    main()
