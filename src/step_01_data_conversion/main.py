"""
主函数，整个程序的入口
"""

# import pandas as pd
# import numpy as np
# import re

import torch
# import pytorch_forecasting as ptf
# import pytorch_lightning as ptl

from utils.data_pre_processor import BatteryDataPreProcessor

if __name__ == '__main__':

    print(torch.__version__)
    # print(torch.cuda.is_available())
    # print(ptf.__version__)

    processor = BatteryDataPreProcessor(r"data/raw/battery_aging_cycle_data.xlsx",
                                        r"data/processed/battery_aging_cycle_data.csv")
    processor.process()
    processor.save()