import logging
import os
import sys
from typing import List

from config.config_loader import (
    collect_yaml_files_from_dir,
    find_config_dir,
    find_project_root_dir,
    load_and_merge_configs,
)
from config.config_setter import setup_logging
from src.pipeline.preprocess_pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

# 定义用于查找项目根目录的标记文件
PROJECT_ROOT_MARKER = "pyproject.toml"
# 定义配置目录在项目根目录下的相对路径
CONFIG_DIR_RELATIVE_PATH = "config"

# 定义日志配置文件名
LOG_CONFIG_PATH = "logging.yaml"
# 定义文件路径配置文件名
DATA_IO_PATH_CONFIG_PATH = "data_io_paths.yaml"
# 定义数据预处理配置目录名
DATA_PREPROCESS_CONFIG_DIRNAME = "data_preprocess"
# 定义绘图配置目录名
PLOT_CONFIG_DIRNAME = "plot"


if __name__ == "__main__":
    logger.info("数据预处理开始, 开始加载和处理配置...")

    project_root_dir = find_project_root_dir(marker_file=PROJECT_ROOT_MARKER)
    logger.info(f"检测到的项目根目录: {project_root_dir}")
    config_dir = find_config_dir(
        project_root_dir=project_root_dir, config_dir_relative_path=CONFIG_DIR_RELATIVE_PATH
    )
    logger.info(f"检测到的配置文件目录: {config_dir}")

    log_config_path = os.path.join(config_dir, LOG_CONFIG_PATH)
    data_io_path_config_path = os.path.join(config_dir, DATA_IO_PATH_CONFIG_PATH)
    data_preprocess_config_dir = os.path.join(config_dir, DATA_PREPROCESS_CONFIG_DIRNAME)
    plot_config_dir = os.path.join(config_dir, PLOT_CONFIG_DIRNAME)

    # 配置日志
    setup_logging(log_config_path)

    all_config_files: List[str] = []
    all_config_files.append(data_io_path_config_path)
    all_config_files = collect_yaml_files_from_dir(data_preprocess_config_dir, all_config_files)
    all_config_files = collect_yaml_files_from_dir(plot_config_dir, all_config_files)

    # 检查是否有任何配置文件被收集到
    if not all_config_files:
        logger.error(
            "错误: 未找到任何配置文件, 以及指定配置目录下的 *.yaml 文件). 基础配置缺失. 流程终止."
        )
        sys.exit(1)
    else:
        logger.info(f"已收集 {len(all_config_files)} 个配置文件路径.")

    logger.info("开始加载并合并所有配置文件...")
    config = load_and_merge_configs(all_config_files)
    logger.info("所有配置文件加载并合并完成. 最终配置已准备好.")

    # 实例化并运行 Pipeline
    pipeline = PreprocessingPipeline(config, project_root_dir)
    pipeline.run()

    logger.info("数据预处理流程执行完毕.")
