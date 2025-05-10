import os
import pandas as pd

from modules.data_pre_process.data_cleaner import DataCleaner
from modules.data_pre_process.data_imputer import DataImputer
from modules.data_pre_process.data_denoiser import DataDenoiser
from modules.data_pre_process.data_decomposer import DataDecomposer

from modules.visualization.plotter import Plotter
from modules.visualization.style_setter import StyleSetter

if __name__ == '__main__':
    # 绘图常量
    PLOT_SIZE_CM = (16, 11)  # 坐标轴区域大小，单位厘米（宽，高）
    PLOT_DPI = 300           # 图片分辨率

    # 数据路径
    CSV_PATH = ('D:/OneDrive/Project/Tertiary_Edu/Bachelor\'s/Culmination Design/Codes/'
                'src/step_01_data_conversion/data/processed/battery_aging_cycle_data.csv')

    # 输出路径
    PROCESSED_DATA_DIR = 'data/processed'
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    CLEANED_PATH = os.path.join(PROCESSED_DATA_DIR, 'step_0_battery_aging_cycle_data_cleaned.csv')
    ANOMALIES_PATH = os.path.join(PROCESSED_DATA_DIR, 'step_0_battery_aging_cycle_data_anomalies.csv')
    IMPUTED_PATH = os.path.join(PROCESSED_DATA_DIR, 'step_1_battery_aging_cycle_data_imputed.csv')
    DENOISED_PATH = os.path.join(PROCESSED_DATA_DIR, 'step_2_battery_aging_cycle_data_denoised.csv')
    DECOMPOSED_PATH = os.path.join(PROCESSED_DATA_DIR, 'step_3_battery_aging_cycle_data_decomposed.csv')

    PLOT_DIR = 'plots'

    # 读取数据
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"成功读取数据，形状: {df.shape}")
    except FileNotFoundError:
        print(f"错误：文件 {CSV_PATH} 不存在")
        exit(1)

    # 确认必要列
    required_columns = ['battery_id', 'target', 'cycle_idx', 'charge_rate', 'temperature', 'pressure', 'dod']
    if not all(col in df.columns for col in required_columns):
        print(f"错误：数据缺少必要列 {required_columns}")
        exit(1)

    """
    step1: 数据清洗
    """
    # 初始清洗
    df = df.dropna(subset=required_columns)
    df = df[df['target'] > 22]
    df = df[df['cycle_idx'] > 0]

    # 孤立森林清洗
    features = ['target', 'cycle_idx']
    cleaner = DataCleaner(features, contamination=0.03, random_state=42)
    df_cleaned, df_with_anomaly = cleaner.clean_with_isolation_forest(df)

    # 提取异常数据
    df_anomalies = df_with_anomaly[df_with_anomaly['anomaly'] == -1].drop(columns=['anomaly'])

    # 打印结果
    print(f"清洗后数据形状: {df_cleaned.shape}")
    print(f"检测到异常点数量: {len(df_anomalies)}")

    # 保存清洗数据
    df_cleaned.to_csv(CLEANED_PATH, index=False)
    print(f"清洗后的数据已保存至: {CLEANED_PATH}")
    if not df_anomalies.empty:
        df_anomalies.to_csv(ANOMALIES_PATH, index=False)
        print(f"异常数据已保存至: {ANOMALIES_PATH}")
    else:
        print("未检测到异常数据")

    """
    step2:缺失值处理
    """
    imputer = DataImputer(target_column='target')
    df_imputed = imputer.impute_missing_values(df_cleaned)

    # 检查缺失值处理结果
    print(f"缺失值处理后数据形状: {df_imputed.shape}")
    print(f"剩余缺失值数量: {df_imputed['target'].isna().sum()}")

    # 保存插值数据
    df_imputed.to_csv(IMPUTED_PATH, index=False)
    print(f"插值后的数据已保存至: {IMPUTED_PATH}")

    """
    step3: 数据去噪
    """
    denoiser = DataDenoiser(target_column='target', wavelet='db4', threshold_mode='soft')
    df_denoised = denoiser.denoise_data(df_imputed)

    # 检查去噪结果
    print(f"去噪后数据形状: {df_denoised.shape}")
    print(f"去噪列 'target_denoised' 是否有缺失值: {df_denoised['target_denoised'].isna().sum()}")

    # 保存去噪数据
    df_denoised.to_csv(DENOISED_PATH, index=False)
    print(f"去噪后的数据已保存至: {DENOISED_PATH}")

    """
    step4: 趋势分解
    """
    decomposer = DataDecomposer(
        target_column='target_denoised',
        K=7,  # 增加模态数
        alpha=2000,  # 降低带宽约束
        trend_modes=4
    )
    df_decomposed = decomposer.decompose_data(df_denoised)

    # 检查分解结果
    print(f"趋势分解后数据形状: {df_decomposed.shape}")
    print(f"趋势列 'target_trend' 是否有缺失值: {df_decomposed['target_trend'].isna().sum()}")
    print(f"残差列 'target_residual' 是否有缺失值: {df_decomposed['target_residual'].isna().sum()}")

    # 保存分解数据
    df_decomposed.to_csv(DECOMPOSED_PATH, index=False)
    print(f"趋势分解后的数据已保存至: {DECOMPOSED_PATH}")

    # 设置绘图风格
    style_setter = StyleSetter()

    # 绘图
    plotter = Plotter(style_setter, PLOT_SIZE_CM, PLOT_DPI)
    """
    step0: 绘制原始数据图像
    """
    plotter.plot_per_battery_raw(df, PLOT_DIR)

    """
    step1: 绘制清洗相关图像
    """
    plotter.plot_per_battery_comparison(df, df_with_anomaly, PLOT_DIR)
    plotter.plot_per_battery_cleaned(df_cleaned, PLOT_DIR)
    plotter.plot_overall_sequence(df, '清洗前总体序列',
                                  os.path.join(PLOT_DIR, 'overall_before_cleaning.png'))
    plotter.plot_overall_sequence(df_cleaned, '清洗后总体序列',
                                  os.path.join(PLOT_DIR, 'overall_after_cleaning.png'))

    """
    step2: 绘制缺失值处理相关图像
    """
    plotter.plot_per_battery_imputed(df_cleaned, df_imputed, PLOT_DIR)  # 新增
    plotter.plot_overall_sequence(df_imputed, '插值后总体序列',
                                  os.path.join(PLOT_DIR, 'overall_after_imputation.png'))

    """
    step3: 绘制去噪相关图像
    """
    plotter.plot_per_battery_denoised(df_denoised, PLOT_DIR)

    """
    step4: 绘制趋势分解相关图像
    """
    plotter.plot_per_battery_decomposed(df_decomposed, PLOT_DIR, show_modes=False)
