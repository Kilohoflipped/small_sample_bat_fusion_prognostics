import logging.config
import sys

from .config_loader import load_yaml_config


def setup_logging(logging_config_path: str) -> None:
    """
    加载指定的日志配置文件 (YAML格式) 并配置应用程序的日志系统

    Args:
        logging_config_path (str): 日志 YAML 配置文件的路径

    Raises:
        FileNotFoundError: 如果指定的日志配置文件不存在
        IOError: 如果读取日志配置文件时发生IO错误
        yaml.YAMLError: 如果解析日志配置文件内容时发生YAML格式错误
        ValueError: 如果加载的配置字典结构不符合logging.config.dictConfig的要求
        TypeError: 如果加载的配置字典中的值类型不正确
    """
    try:
        logging_config = load_yaml_config(logging_config_path)

        # 检查加载的配置是否为字典类型
        if not isinstance(logging_config, dict):
            # 如果加载的配置不是字典, 抛出类型错误
            raise TypeError(f"日志配置文件 {logging_config_path} 加载的内容不是有效的字典格式.")

        # 使用加载的字典配置日志系统.
        # dictConfig 在配置结构无效时可能会抛出 ValueError 或 TypeError.
        logging.config.dictConfig(logging_config)

    except FileNotFoundError as e:
        print(f"错误: 日志配置文件 '{logging_config_path}' 不存在: {e}", file=sys.stderr)
        raise

    except (ValueError, TypeError) as e:
        # 捕获 logging.config.dictConfig 抛出的配置结构或类型错误
        print(f"错误: 应用日志配置失败, 配置结构或类型无效: {e}", file=sys.stderr)
        raise

    except Exception as e:
        # 捕获 logging.config.dictConfig 或其他应用配置时可能发生的未知异常
        print(f"错误: 应用日志配置时发生未知错误: {e}", file=sys.stderr)
        raise
