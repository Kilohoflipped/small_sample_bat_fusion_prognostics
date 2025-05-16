import logging
import os
import sys
from typing import Any, Dict, List

from config.config_loader import (
    collect_yaml_files_from_dir,
    find_project_root_dir,
    load_yaml_config,
    recursive_merge_configs,
)
from config.config_setter import setup_logging
from src.modules.visualization.plotter.preprocess_paper import PreprocessPaperPlotter
from src.modules.visualization.style_setter import StyleSetter

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # --- 步骤 1: 配置日志, 收集配置文件路径 ---
    logger.info("数据预处理开始, 开始加载和处理配置...")

    # 定义用于查找项目根目录的标记文件
    PROJECT_ROOT_MARKER = "pyproject.toml"
    # 定义配置目录在项目根目录下的相对路径
    CONFIG_DIR_RELATIVE_PATH = "config"

    # 定义日志配置文件名
    LOG_CONFIG_PATH = "logging.yaml"
    # 定义文件路径配置文件名
    DATA_IO_PATH_CONFIG_PATH = "data_io_paths.yaml"
    # 定义绘图配置目录名
    PLOT_CONFIG_DIRNAME = "plot"

    try:
        # 从当前脚本执行的目录开始向上查找项目根目录
        project_root = find_project_root_dir(marker_file=PROJECT_ROOT_MARKER)
        logger.info(f"检测到的项目根目录: {project_root}")

        # 构建总配置目录的绝对路径
        config_dir = os.path.join(project_root, CONFIG_DIR_RELATIVE_PATH)
        if not os.path.isdir(config_dir):
            logger.error(f"配置目录不存在: {config_dir}. 请检查项目结构.")
            sys.exit(1)
        logger.info(f"配置目录路径: {config_dir}")

        # 构建重要配置路径
        log_config_path = os.path.join(config_dir, LOG_CONFIG_PATH)
        data_io_path_config_path = os.path.join(config_dir, DATA_IO_PATH_CONFIG_PATH)
        plot_config_dir = os.path.join(config_dir, PLOT_CONFIG_DIRNAME)

    except FileNotFoundError as e:
        # 如果 find_project_root_dir 没有找到标记文件
        logger.error(f"查找项目根目录失败: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"查找项目路径时发生未知错误: {e}")
        sys.exit(1)
    # 配置日志
    setup_logging(log_config_path)

    # --- 步骤 2: 收集所有需要加载的配置文件路径 ---
    logger.info("开始收集所有需要加载的配置文件路径...")
    all_config_files: List[str] = []

    logger.info(f"收集关键文件: {data_io_path_config_path}")
    all_config_files.append(data_io_path_config_path)

    # 收集绘图配置目录下的文件 (如果目录存在且是目录)
    logger.info(f"收集绘图配置目录文件: {plot_config_dir}")
    all_config_files = collect_yaml_files_from_dir(plot_config_dir, all_config_files)
    # 检查是否有任何配置文件被收集到
    if not all_config_files:
        logger.error(
            f"错误: 未找到任何配置文件, 以及指定配置目录下的 *.yaml 文件). 基础配置缺失. 流程终止."
        )
        sys.exit(1)
    else:
        logger.info(f"已收集 {len(all_config_files)} 个配置文件路径.")

    # --- 步骤 3: 初始化总配置字典并加载合并所有文件 ---
    # 从一个空字典开始合并, 后续文件的内容会覆盖前面文件的相同 key
    config: Dict[str, Any] = {}

    logger.info("开始加载并合并所有配置文件...")
    # 遍历所有收集到的文件路径, 逐个加载和合并, 任何失败都终止程序
    for file_path in all_config_files:
        try:
            logger.info(f"正在处理文件: {file_path}")
            # 使用 load_yaml_config 函数加载单个文件
            sub_config = load_yaml_config(file_path)

            # 使用 recursive_merge_configs 函数合并到总配置中
            config = recursive_merge_configs(config, sub_config)
            logger.info(f"文件 '{file_path}' 加载并合并成功.")

        except Exception as e:
            logger.exception(f"错误: 处理配置文件 '{file_path}' 时发生错误.")
            sys.exit(1)

    logger.info("所有配置文件加载并合并完成. 最终配置已准备好.")

    # 使用自定义风格
    style_setter = StyleSetter(config=config.get("plot_style"))

    # 2. 读取csv目录
    csv_path = os.path.join(
        project_root,
        config.get("output_dirs", {}).get("preprocessed_data"),
        "step_1_battery_aging_cycle_data_cleaned.csv",
    )

    # 3. 指定输出目录
    plot_output_dir = os.path.join(
        project_root,
        config.get("output_dirs", {}).get("plots"),
        "preprocess_paper",
    )

    # 4. 创建 PreprocessPaperPlotter 实例
    try:
        paper_plotter = PreprocessPaperPlotter(
            style_setter=style_setter, csv_path=csv_path, output_dir=plot_output_dir
        )

        # 5. 指定噪声展示的 cycle_idx 范围，并调用绘图方法
        noise_cycle_range = (110, 125)
        inset_location = "lower center"
        paper_plotter.plot_per_battery_paper_style(noise_cycle_range, inset_location)

    except (FileNotFoundError, ValueError, Exception) as e:
        logger.error(f"绘图过程中发生错误: {e}")
