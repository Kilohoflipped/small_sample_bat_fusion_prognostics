"""
主函数，将电池老化数据格式转换为长格式
"""

from utils.data_preprocessor import BatteryDataPreprocessor

if __name__ == '__main__':
    preprocessor = BatteryDataPreprocessor(r"data/raw/battery_aging_cycle_data.xlsx",
                                           r"data/processed/battery_aging_cycle_data.csv")
    preprocessor.run_preprocessing()
    preprocessor.save_preprocessed_data()
