import os
import yaml
import logging
from typing import Dict, Any, Optional, List
import sys
import copy

logger = logging.getLogger(__name__)


def find_project_root(start_path: Optional[str] = None,
                      marker_file: str = "pyproject.toml") -> str:
    """
    从给定的路径或当前执行脚本的目录向上查找项目根目录
    通过查找特定的标记文件实现

    Args:
        start_path (str, optional): 开始查找的路径. 如果为 None 则从当前执行脚本的目录开始
        marker_file (str): 用于标识项目根目录的文件名(默认为"pyproject.toml", 此外还可以使用".git"等)

    Returns:
        str: 项目根目录的绝对路径

    Raises:
        FileNotFoundError: 如果在到达文件系统根目录之前没有找到标记文件
    """
    if start_path is None:
        # 使用 sys.argv[0] 获取执行脚本的路径
        # os.path.abspath 确保是绝对路径
        # os.path.dirname 获取所在的目录
        initial_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    else:
        initial_dir = os.path.abspath(start_path)

    # 向上查找, 直到找到标记文件或到达文件系统根目录
    current_dir = initial_dir
    while True:
        marker_path = os.path.join(current_dir, marker_file)
        if os.path.exists(marker_path):
            logger.debug(f"找到项目根目录: {current_dir} (通过标记文件: {marker_file})")
            return current_dir

        # 移动到父目录
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            # 到达文件系统根目录, 停止查找
            break
        current_dir = parent_dir

    # 如果没找到, 抛出异常
    raise FileNotFoundError(
        f"未能在 '{initial_dir}' "
        f"向上查找父目录时找到标记文件: '{marker_file}' "
        "请确保在项目根目录或其子目录中运行脚本, 并且根目录包含该标记."
    )


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件.

    Args:
        config_path (str): 配置文件的路径

    Returns:
        Dict[str, Any]: 加载的配置字典

    Raises:
        FileNotFoundError: 如果配置文件不存在
        yaml.YAMLError: 如果配置文件不是有效的 YAML 格式
        IOError: 如果读取文件时发生其他输入/输出错误 (例如权限问题)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.exception(f"加载配置文件 '{config_path}' 时发生 YAML 格式错误: {e}")
        raise
    except IOError as e:
        logger.exception(f"读取配置文件 '{config_path}' 时发生 IO 错误: {e}")
        raise


def recursive_merge_configs(base: Dict[str, Any], head: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并两个配置字典

    如果键在两个字典中都存在且都是字典, 则递归合并
    否则, head 中的值会覆盖 base 中的值
    在覆盖时, 对 head 中的值进行深拷贝, 以避免对原始 head 字典的修改影响合并后的结果
    主要是对于值为可变类型的情况

    Args:
        base: 作为基础的配置字典
        head: 包含待合并内容的配置字典, 其值会覆盖 base 中的同名键

    Returns:
        一个新的字典, 包含合并后的结果
    """
    merged = base.copy()
    for k, v in head.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = recursive_merge_configs(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def collect_yaml_files_from_dir(directory_path: str,
                                all_yaml_files_list: List[str],
                                ) -> List[str]:
    """
    检查给定的路径是否是目录, 如果是, 收集其中所有 .yaml 文件的路径并添加到列表中
    处理目录读取错误 (如权限问题) 为致命错误

    Args:
        directory_path (str): 需要检查和收集文件的目录路径
        all_yaml_files_list (List[str]): 收集到的 YAML 文件路径将添加到此列表中
    """
    all_yaml_files_list_collected = all_yaml_files_list

    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        logger.info(f"收集配置目录文件: {directory_path}")
        try:
            filenames = os.listdir(directory_path)
            for filename in filenames:
                if filename.endswith('.yaml'):
                    file_path = os.path.join(directory_path, filename)
                    all_yaml_files_list_collected.append(file_path)
                    logger.debug(f"已收集文件: {file_path}")
        except IOError as e:
            # 捕获读取目录列表时的错误 (如权限)
            logger.exception(f"错误: 读取配置目录 '{directory_path}' 时发生IO错误.")
            raise
    else:
        # 如果目录不存在或不是目录, 记录信息
        logger.info(
            f"配置目录不存在或不是目录: {directory_path}. 未收集该目录下的文件."
        )

    return all_yaml_files_list_collected
