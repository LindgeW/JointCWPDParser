import logging
import sys
import os
import json


# 自定义日志模块
class MyLogger(object):
    # 日志级别关系映射
    level_dict = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, log_conf):
        self.conf_opts = MyLogger.load_config(log_conf)

    @staticmethod
    def load_config(config_path):
        assert os.path.exists(config_path)
        with open(config_path, 'r', encoding='utf-8') as fin:
            opts = json.load(fin)
        return opts

    def get_logger(self, log_path=None):
        if log_path is None:
            log_path = self.conf_opts['default_log_path']

        # 设置日志输出格式
        fmt = logging.Formatter(fmt=self.conf_opts['log_fmt'],
                                datefmt=self.conf_opts['date_fmt'])
        # 创建一个名为filename的日志器
        logger = logging.getLogger(log_path)
        # 设置日志级别
        logger.setLevel(self.level_dict[self.conf_opts['log_level']])

        if self.conf_opts['to_console']:
            # 获取控制台输出的处理器
            console_handler = logging.StreamHandler(sys.stdout)  # 默认是sys.stderr
            # 设置控制台处理器的等级为DEBUG
            console_handler.setLevel(self.level_dict['info'])
            # 设置控制台输出日志的格式
            console_handler.setFormatter(fmt)
            logger.addHandler(console_handler)

        if self.conf_opts['to_file']:
            # 获取路径的目录
            log_dir = os.path.dirname(log_path)
            if os.path.isdir(log_dir) and not os.path.exists(log_dir):
                # 目录不存在则创建
                os.makedirs(log_dir)

            # 获取文件输出的处理器
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            # 设置文件输出处理器的等级为INFO
            file_handler.setLevel(self.level_dict['debug'])
            # 设置文件输出日志的格式
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

        return logger


if __name__ == '__main__':
    logger = MyLogger('log_config.json').get_logger(log_path='../log/xx.log')
    logger.info('正在执行。。。')
    print(__name__)
