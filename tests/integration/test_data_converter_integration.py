import pathlib

import pandas as pd
import pytest

from src.modules.data_preprocess.data_converter import DataConverter


# --- Fixture: 获取测试数据文件路径 ---


@pytest.fixture
def sample_xlsx_path(request):
    # 获取当前测试文件 (test_converter_integration.py) 的所在目录的 Path 对象
    # __file__ 是当前文件路径，parent 是其父目录 (tests/integration/)
    test_file_dir = pathlib.Path(request.fspath).parent
    # 从 tests/integration/ 退回到 tests/ (用 '..')，然后进入 test_data/ 找到文件
    data_file_path = test_file_dir / ".." / "test_data" / "battery_aging_cycle_data.xlsx"

    # 确保文件存在，否则跳过测试
    if not data_file_path.exists():
        pytest.skip(f"测试数据文件不存在: {data_file_path}")

    return str(data_file_path)  # load_and_convert 接收字符串路径


# --- Fixture: 获取并创建测试结果目录 ---


@pytest.fixture
def test_results_dir(request):
    # 获取当前测试文件所在的目录 (tests/integration/)
    test_file_dir = pathlib.Path(request.fspath).parent
    # 构建测试结果目录路径: 从 tests/integration/ 返回一层到 tests/，然后进入 test_results/
    results_dir = test_file_dir / ".." / "test_results"

    # 如果目录不存在，则创建它
    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir  # 返回 Path 对象


# --- 集成测试用例 ---


def test_convert_and_save_sample_file(sample_xlsx_path, test_results_dir):
    """
    测试 load_and_convert 方法使用 tests/test_data/1111.xlsx 文件，并保存结果到 tests/test_results/
    """
    converter = DataConverter()

    # 调用要测试的方法
    processed_df = converter.load_and_convert(sample_xlsx_path)

    # --- 核心测试断言：验证转换结果是否符合预期 ---
    # 1. 检查返回的是否是 DataFrame
    assert isinstance(processed_df, pd.DataFrame)

    # 2. 检查 DataFrame 是否非空
    # 如果文件可能导致转换失败或结果为空，这里的断言需要相应调整
    assert (
        not processed_df.empty
    ), f"转换结果是空的 DataFrame，检查文件内容和解析逻辑是否有问题：{sample_xlsx_path}"

    # 3. 检查是否包含预期的关键列
    expected_cols = ["cycle_idx", "target", "battery_id"]
    assert all(
        col in processed_df.columns for col in expected_cols
    ), "转换后的 DataFrame 缺少预期的关键列"

    # --- 保存结果到 CSV（这是测试的副作用，用于人工检查） ---
    if not processed_df.empty:
        # 构建输出 CSV 文件的完整路径
        output_csv_path = test_results_dir / "battery_aging_cycle_data_processed.csv"
        # 保存 DataFrame 到 CSV，不保存索引
        processed_df.to_csv(output_csv_path, index=False)
        print(f"\n转换结果已保存至: {output_csv_path}")
    else:
        print(f"\n转换结果为空，未保存 CSV 文件")
