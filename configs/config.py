import sys

from yacs.config import CfgNode as _CfgNode
# 导入 sys 模块用于修改模块搜索路径，导入 yacs.config 中的 CfgNode 类，它是一个配置节点，用于存储配置信息。

sys.path.append('..')
# 将上级目录添加到模块搜索路径，这通常是为了方便导入上级目录中的模块。


class CfgNode(_CfgNode):
    # 定义了 CfgNode 类，它继承自 yacs.config 中的 _CfgNode 类。
    def setup(self, args):
        # 定义了 setup 方法，用于从文件和命令行参数初始化配置。
        if args.config_file is not None:
            self.merge_from_file((args.config_file))
            # 如果提供了配置文件，则将配置节点与文件中的配置合并。
        if args.opts is not None:
            self.merge_from_list(args.opts)
            # 如果提供了命令行选项，则将配置节点与这些选项合并。

    def __str__(self):
        # 定义了 __str__ 方法，用于返回配置节点的字符串表示形式，这在打印配置时很有用。
        def _indent(s_, num_spaces):
            # 定义了一个内部函数 _indent，用于在字符串的每一行前面添加空格进行缩进。
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        # 初始化两个字符串变量，r 用于最终结果，s 用于存储每一行的配置字符串。
        for k, v in sorted(self.items()):
            # 遍历配置节点中的所有项，sorted 确保字典是按键排序的。
            seperator = "\n" if isinstance(v, CfgNode) else " "
            # 根据值的类型确定分隔符，如果值是另一个 CfgNode，则分隔符为换行符，否则为一个空格。
            v = f"'{v}'" if isinstance(v, str) else v
            # 如果是字符串类型，则在字符串周围添加引号。
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            # 格式化配置项的字符串表示形式。
            attr_str = _indent(attr_str, 4)
            # 对属性字符串进行缩进。
            s.append(attr_str)
            # 将格式化后的字符串添加到列表 s 中。
        r += "\n".join(s)
        # 将所有配置项的字符串连接成一个单一的字符串，配置项之间用换行符分隔。
        return r
        # 返回配置的字符串表示。

global_cfg = CfgNode()
# 创建一个全局配置节点实例。
CN = CfgNode
# 为 CfgNode 类设置一个简短的别名 CN。


def get_cfg():
    '''
    Get a copy of the default config.

    Returns:
        a CfgNode instance.
    '''
    # 定义一个函数 get_cfg，用于获取默认配置的副本。
    from .default import _C
    # 从 default 模块导入默认配置节点 _C。
    return _C.clone()
    # 返回默认配置节点的副本。
