import logging
import os
import sys
from typing import List

import pandas as pd

from config.config_loader import (
    collect_yaml_files_from_dir,
    find_config_dir,
    find_project_root_dir,
    load_and_merge_configs,
)
from config.config_setter import setup_logging
from src.modules.visualization.plotter.preprocess import PreprocessPlotter
from src.modules.visualization.style_setter import StyleSetter

# --- 主要执行部分 ---
if __name__ == "__main__":
    # 定义用于查找项目根目录的标记文件
    PROJECT_ROOT_MARKER = "pyproject.toml"
    # 定义配置目录在项目根目录下的相对路径
    CONFIG_DIR_RELATIVE_PATH = "config"

    # 定义日志配置文件名
    LOG_CONFIG_FILENAME = "logging.yaml"
    # 定义文件路径配置文件名
    DATA_IO_PATH_CONFIG_FILENAME = "data_io_paths.yaml"
    # 定义绘图配置目录名
    PLOT_CONFIG_DIRNAME = "plot"

    # --- 模拟日志和配置加载 ---
    print("--- 模拟日志和配置加载 ---")
    project_root_dir = find_project_root_dir(marker_file=PROJECT_ROOT_MARKER)
    config_dir = find_config_dir(
        project_root_dir=project_root_dir, config_dir_relative_path=CONFIG_DIR_RELATIVE_PATH
    )

    log_config_path = os.path.join(config_dir, LOG_CONFIG_FILENAME)
    data_io_path_config_path = os.path.join(config_dir, DATA_IO_PATH_CONFIG_FILENAME)
    plot_config_dir = os.path.join(config_dir, PLOT_CONFIG_DIRNAME)

    # 配置日志
    setup_logging(log_config_path)
    logger = logging.getLogger(__name__)  # 获取配置后的 logger

    logger.info("模拟配置加载中...")

    all_config_files: List[str] = []
    # 添加 data_io_paths.yaml，即使可能不存在，模拟路径仍然可以存在于合并配置中
    all_config_files.append(data_io_path_config_path)
    # 尝试收集 plot 目录下的所有 yaml 文件
    all_config_files = collect_yaml_files_from_dir(plot_config_dir, all_config_files)

    if not all_config_files:
        logger.error("错误: 未找到任何配置文件用于加载 StyleSetter 配置. 绘图流程终止.")
        sys.exit(1)

    config = load_and_merge_configs(all_config_files)
    logger.info("配置加载完成.")

    # --- 模拟加载数据和绘图 ---
    logger.info("--- 模拟加载数据和绘图 ---")

    plot_data_paths: str = config.get("output_dirs", {}).get("preprocessed_data", "")
    plot_data_paths = os.path.join(project_root_dir, plot_data_paths)
    plot_output_dir: str = config.get("output_dirs", {}).get("plots", "plots")
    plot_output_dir = os.path.join(project_root_dir, plot_output_dir)

    raw_data_csv_path = os.path.join(
        plot_data_paths, "step0_battery_aging_cycle_data_converted.csv"
    )
    anomaly_data_csv_path = os.path.join(
        plot_data_paths, "step1_battery_aging_cycle_data_anomalies.csv"
    )
    cleaned_data_csv_path = os.path.join(
        plot_data_paths, "step1_battery_aging_cycle_data_cleaned.csv"
    )

    imputation_original_csv_path = os.path.join(
        plot_data_paths, "step1_battery_aging_cycle_data_cleaned.csv"
    )
    imputation_imputed_csv_path = os.path.join(
        plot_data_paths, "step2_battery_aging_cycle_data_imputed.csv"
    )

    denoising_original_csv_path = os.path.join(
        plot_data_paths, "step2_battery_aging_cycle_data_imputed.csv"
    )
    denoising_denoised_csv_path = os.path.join(
        plot_data_paths, "step3_battery_aging_cycle_data_denoised.csv"
    )

    standardized_data_csv_path = os.path.join(
        plot_data_paths, "step4_battery_aging_cycle_data_standardized.csv"
    )

    plot_output_dir = os.path.join(plot_output_dir, "preprocess")
    os.makedirs(plot_output_dir, exist_ok=True)
    logger.info(f"图表将保存至: {plot_output_dir}")

    # 实例化 StyleSetter 和 PreprocessPlotter
    plot_style_config = config.get("plot_style", {})  # 从合并配置中获取 plot_style 部分
    style_setter = StyleSetter(plot_style_config)
    plotter = PreprocessPlotter(style_setter)

    # 定义目标列名
    target_column_name = "target"

    # 绘制原始数据图
    try:
        df_raw = pd.read_csv(raw_data_csv_path)
        logger.info(f"加载原始数据用于绘图: {raw_data_csv_path}")
        plotter.plot_per_battery_raw(df_raw, plot_output_dir, target_column=target_column_name)
    except FileNotFoundError:
        logger.warning(f"原始数据 CSV 未找到: {raw_data_csv_path}, 跳过绘制原始数据图.")
    except Exception as e:
        logger.error(f"绘制原始数据图时发生错误: {e}")

    # 绘制原始数据与异常点对比图
    try:
        df_original_for_anomaly = pd.read_csv(raw_data_csv_path)  # 假设原始数据仍然需要
        df_with_anomaly = pd.read_csv(anomaly_data_csv_path)  # 包含 anomaly 列
        logger.info(f"加载数据用于绘制异常点对比图: {raw_data_csv_path}, {anomaly_data_csv_path}")
        plotter.plot_per_battery_comparison(
            df_original_for_anomaly,
            df_with_anomaly,
            plot_output_dir,
            target_column=target_column_name,
        )
    except FileNotFoundError:
        logger.warning(
            f"用于异常点对比的 CSV 未找到: {raw_data_csv_path} 或 {anomaly_data_csv_path}, 跳过绘制异常点对比图."
        )
    except Exception as e:
        logger.error(f"绘制异常点对比图时发生错误: {e}")

    # 绘制清洗后数据图
    try:
        df_cleaned = pd.read_csv(cleaned_data_csv_path)
        logger.info(f"加载清洗后数据用于绘图: {cleaned_data_csv_path}")
        plotter.plot_per_battery_cleaned(
            df_cleaned, plot_output_dir, target_column=target_column_name
        )
    except FileNotFoundError:
        logger.warning(f"清洗后数据 CSV 未找到: {cleaned_data_csv_path}, 跳过绘制清洗后数据图.")
    except Exception as e:
        logger.error(f"绘制清洗后数据图时发生错误: {e}")

    # 绘制插值前与插值后数据对比图
    try:
        df_impute_original = pd.read_csv(imputation_original_csv_path)
        df_impute_imputed = pd.read_csv(imputation_imputed_csv_path)
        logger.info(
            f"加载数据用于绘制插值对比图: {imputation_original_csv_path}, {imputation_imputed_csv_path}"
        )
        plotter.plot_per_battery_imputed(
            df_impute_original,
            df_impute_imputed,
            plot_output_dir,
            original_column=target_column_name,
            imputed_column=target_column_name,
        )
    except FileNotFoundError:
        logger.warning(
            f"用于插值对比的 CSV 未找到: {imputation_original_csv_path} 或 {imputation_imputed_csv_path}, 跳过绘制插值对比图."
        )
    except Exception as e:
        logger.error(f"绘制插值对比图时发生错误: {e}")

    # 绘制去噪前与去噪后数据对比图
    try:
        df_denoise_original = pd.read_csv(denoising_original_csv_path)
        df_denoise_denoised = pd.read_csv(denoising_denoised_csv_path)
        logger.info(
            f"加载数据用于绘制去噪对比图: {denoising_original_csv_path}, {denoising_denoised_csv_path}"
        )
        plotter.plot_per_battery_denoised(
            df_denoise_original,
            df_denoise_denoised,
            plot_output_dir,
            original_column=target_column_name,
            denoised_column=target_column_name,
        )  # 假设去噪是基于插值后的数据
    except FileNotFoundError:
        logger.warning(
            f"用于去噪对比的 CSV 未找到: {denoising_original_csv_path} 或 {denoising_denoised_csv_path}, 跳过绘制去噪对比图."
        )
    except Exception as e:
        logger.error(f"绘制去噪对比图时发生错误: {e}")

    # 绘制标准化后数据折线图
    try:
        df_standardized = pd.read_csv(standardized_data_csv_path)
        logger.info(f"加载标准化后数据用于绘图: {standardized_data_csv_path}")
        plotter.plot_per_battery_standardized(
            df_standardized,
            plot_output_dir,
            standardized_column=target_column_name,
        )  # 假设标准化后的列名
    except FileNotFoundError:
        logger.warning(
            f"标准化后数据 CSV 未找到: {standardized_data_csv_path}, 跳过绘制标准化数据图."
        )
    except Exception as e:
        logger.error(f"绘制标准化数据图时发生错误: {e}")

    logger.info("绘图流程模拟执行完毕.")
