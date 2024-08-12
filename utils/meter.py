import logging
from copy import deepcopy


class MaxMeter(object):
    """
    Keeps track of the max of all the values that are 'add'ed
    """
    # MaxMeter 类用于追踪添加到其中的所有值的最大值。
    def __init__(self):
        self.max = None
        # 构造函数初始化 max 属性为 None，表示目前还没有值被添加。

    def update(self, value):
        """
        Add a value to the accumulator.
        :return: `true` if the provided value became the new max
        """
        # update 方法用于添加一个值，并检查这个值是否是新的 max 值。
        # 如果是，更新 max 属性并返回 True，否则返回 False。
        # 这里使用了 deepcopy 来确保值是独立存储的，但在这种情况下可能不是必需的，因为通常数字类型的比较可以直接使用等值比较而不需要复制。
        if self.max is None or value > self.max:
            self.max = deepcopy(value)
            return True
        else:
            return False

    def value(self):
        """Access the current running average"""
        # value 方法返回当前记录的最大值。
        return self.max


class MinMeter(object):
    """
    Keeps track of the min of all the values that are 'add'ed
    """
    # MinMeter 类用于追踪添加到其中的所有值的最小值。
    def __init__(self):
        # 构造函数初始化 min 属性为 None。
        self.min = None

    def update(self, value):
        """
        Add a value to the accumulator.
        :return: `true` if the provided value became the new max
        """
        # update 方法用于添加一个值，并检查这个值是否是新的 min 值。更新和返回逻辑与 MaxMeter 类似。
        if self.min is None or value < self.min:
            self.min = deepcopy(value)
            return True
        else:
            return False

    def value(self):
        """Access the current running average"""
        # value 方法返回当前记录的最小值。
        return self.min


class AverageMeter(object):
    """Computes and stores the average and current value"""
    # AverageMeter 类用于计算和存储平均值和当前值。
    def __init__(self):
        # 构造函数调用 reset 方法来初始化统计数据。
        self.reset()

    def reset(self):
        # reset 方法将所有统计数据重置为初始值。
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.max = -float("inf")
        self.min = float("inf")
        self.count = 0

    def update(self, val, n=1):
        # update 方法用于添加一个值（和它的数量 n），更新总和、计数、当前值、平均值、最大值和最小值。
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = val if val > self.max else self.max
        self.min = val if val < self.min else self.min

    def make_summary(self, key="None"):
        # make_summary 方法生成一个包含所有统计数据的摘要字典，使用传入的 key 作为前缀。
        sum_key = key + "/" + "sum"
        count_key = key + "/" + "count"
        avg_key = key + "/" + "avg"
        max_key = key + "/" + "max"
        min_key = key + "/" + "min"
        final_key = key + "/" + "final"
        summary = {
            sum_key: self.sum,
            count_key: self.count,
            avg_key: self.avg,
            max_key: self.max,
            min_key: self.min,
            final_key: self.val,
        }
        return summary









