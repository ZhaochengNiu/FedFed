import os  # 导入os模块，用于操作系统功能，如文件路径操作。
import json  # 导入json模块，用于处理JSON数据。
import time  # 导入time模块，用于时间相关的操作。
import platform  # 导入platform模块，用于获取平台或系统信息。
import logging  # 导入logging模块，用于日志记录。


def logging_config(args, process_id):
    # customize the log format
    while logging.getLogger().handlers:
        logging.getLogger().handlers.clear()  # 清空当前的日志处理器。
    console = logging.StreamHandler()  # 创建一个新的控制台日志处理器。
    if args.level == 'INFO':
        console.setLevel(logging.INFO)  # 如果args指定级别为INFO，设置日志级别为INFO。
    elif args.level == 'DEBUG':
        console.setLevel(logging.DEBUG)  # 如果args指定级别为DEBUG，设置日志级别为DEBUG。
    else:
        raise NotImplementedError   # 如果args指定级别既不是INFO也不是DEBUG，抛出未实现异常。
    formatter = logging.Formatter(str(process_id) +   # 创建日志格式器，包括进程ID。
        ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    console.setFormatter(formatter)   # 将格式器应用到控制台日志处理器。
    # Create an instance
    logging.getLogger().addHandler(console)   # 将控制台日志处理器添加到日志器。
    # logging.getLogger().info("test")
    logging.basicConfig()  # 基本配置日志系统（这里可能存在配置重复的问题）。
    logger = logging.getLogger()   # 获取日志器的实例。
    if args.level == 'INFO':
        logger.setLevel(logging.INFO)   # 设置日志器的日志级别为INFO。
    elif args.level == 'DEBUG':
        logger.setLevel(logging.DEBUG)   # 设置日志器的日志级别为DEBUG。
    else:
        raise NotImplementedError   # 如果args指定级别不是INFO或DEBUG，抛出未实现异常。
    logging.info(args)
    # 记录args的日志信息。


class Logger(object):
    # 定义Logger类，包含日志级别的常量。
    INFO = 0
    DEBUG = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

    @classmethod
    def config_logger(cls, file_folder='.', level="info",
                        save_log=False, display_source=False):
        """
        :param filename: ending with .json
        :param auto_save: save the JSON file after every addition
        """
        # 类方法 config_logger，用于配置日志系统。
        cls.file_folder = file_folder  # 设置日志文件的文件夹。
        cls.file_json = os.path.join(file_folder, "log-1.json")  # 设置JSON日志文件的路径。
        # cls.file_log can be changed by add_log_file()
        cls.file_log = os.path.join(file_folder, "log.log")  # 设置文本日志文件的路径。
        cls.values = []  # 初始化用于缓存日志的列表。
        cls.save_log = save_log  # 设置是否保存日志文件。
        logger = logging.getLogger()  # 获取日志器的实例。
        if display_source:
            cls.formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
            # 设置包含源代码位置的日志格式。
        else:
            cls.formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            # 设置不包含源代码位置的日志格式。
        cls.level = level  # 设置类的日志级别。
        # 根据level参数设置日志器的日志级别。
        if level == "info":
            logger.setLevel(logging.INFO)
        elif level == "debug":
            logger.setLevel(logging.DEBUG)
        elif level == "warning":
            logger.setLevel(logging.WARNING)
        elif level == "error":
            logger.setLevel(logging.ERROR)
        elif level == "critical":
            logger.setLevel(logging.CRITICAL)

        strhdlr = logging.StreamHandler()  # 创建控制台日志处理器。
        strhdlr.setFormatter(cls.formatter)  # 将日志格式器应用到控制台日志处理器。
        logger.addHandler(strhdlr)  # 将控制台日志处理器添加到日志器。
        if save_log:
            cls.add_log_file(cls.file_log)
            # 如果需要保存日志文件，调用add_log_file方法。
        cls.logger = logger # 将配置好的日志器赋值给类属性。


    @classmethod
    def add_log_file(cls, logfile):
        # 类方法 add_log_file，用于添加文件日志处理器。
        assert cls.save_log is True  # 确保cls.save_log为True。
        hdlr = logging.FileHandler(logfile)  # 创建文件日志处理器。
        hdlr.setFormatter(cls.formatter)  # 将日志格式器应用到文件日志处理器。
        cls.logger.addHandler(hdlr)  # 将文件日志处理器添加到日志器。


    @classmethod
    def display_metric(cls, name, values, tags):
        # 类方法display_metric，用于显示度量信息。
        cls.info(
            value="{name} ({tags}): {values} ".format(  # 调用info方法记录格式化的日志信息。
                name=name, values=values)
        )

    @classmethod
    def cache_metric_in_memory(cls, name, values, tags):
        """
        Store a scalar metric. Example:
        name="runtime",
        values={
            "time": current_time,
            "rank": rank,
            "epoch": epoch,
            "best_perf": best_perf,
        },
        tags={"split": "test", "type": "local_model_avg"},
        """
        # # 类方法cache_metric_in_memory，用于将度量信息缓存到内存。
        cls.values.append({"measurement": name, **tags, **values})
        # 将度量信息添加到cls.values列表。

    @classmethod
    def log_timer(cls, name, values, tags):
        # 类方法log_timer，用于记录计时器信息。
        cls.info(
            value="{name} ({tags}): {values} ".format(  # 调用info方法记录格式化的日志信息。
                name=name, values=values)
        )


    @classmethod
    def info(cls, value):
        cls.logger.info(value)  # 使用日志器记录INFO级别的日志信息。

    @classmethod
    def debug(cls, value):
        cls.logger.debug(value)  # 使用日志器记录DEBUG级别的日志信息。

    @classmethod
    def warning(cls, value):
        cls.logger.warning(value)
    
    @classmethod
    def error(cls, value):
        cls.logger.error(value)

    @classmethod
    def critical(cls, value):
        cls.logger.critical(value)


    @classmethod
    def save_json(cls):
        """Save the internal memory to a file."""
        # 类方法save_json，用于将内存中的日志信息保存到JSON文件。
        with open(cls.file_json, "w") as fp:  # 打开JSON文件准备写入。
            json.dump(cls.values, fp, indent=" ")  # 将cls.values列表转换为JSON格式并写入文件。
        if len(cls.values) > 1e3:   # 如果日志条目超过1000条。
            # reset 'values' and redirect the json file to other name.
            cls.values = []  # 重置日志条目列表。
            cls.redirect_new_json()   # 调用redirect_new_json方法重命名JSON文件。


    @classmethod
    def redirect_new_json(cls):
        """get the number of existing json files under the current folder."""
        # # 类方法redirect_new_json，用于重命名JSON文件。
        existing_json_files = [    # 获取当前文件夹下所有的JSON文件。
            file for file in os.listdir(cls.file_folder) if "json" in file
        ]
        cls.file_json = os.path.join(     # 设置新的JSON文件名。
            cls.file_folder, "log-{}.json".format(len(existing_json_files) + 1)
        )


# Usage example
def display_training_stat(conf, tracker, epoch, n_bits_to_transmit):
    # 函数display_training_stat，用于展示训练状态。
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    # 获取当前时间。
    # display the runtime training information.
    Logger.display_metric(  # 调用Logger类的display_metric方法记录训练度量信息。
        name="runtime",
        values={
            "time": current_time,
            "epoch": epoch,
            "n_bits_to_transmit": n_bits_to_transmit / 8 / (2 ** 20),
            # 转换位到兆比特。
            **tracker(),  # 展开tracker()返回的字典。
        },
        tags={"split": "train"}  # 标签为训练。
    )


# Usage example
def display_test_stat(conf, tracker, epoch, label="local"):
    # 函数display_test_stat，用于展示测试状态。
    current_time = time.strftime("%Y-%m-%d %H:%M:%S") # 获取当前时间。
    # display the runtime training information.
    Logger.display_metric(  # 调用 Logger 类的 display_metric 方法记录测试度量信息。
        name="runtime",
        values={
            "time": current_time,
            "epoch": epoch,
            **tracker(),  # 展开tracker()返回的字典。
        },
        tags={"split": "test", "type": label}  # 标签为测试，类型为"local"。
    )
