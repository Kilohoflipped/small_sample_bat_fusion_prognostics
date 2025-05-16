"""
本模块包含用于测试 BatteryStaticFeatures 类中 get_static_feature_names 方法的单元测试.
"""

import dataclasses

from src.modules.data_preprocess.data_converter import BatteryStaticFeatures


# --- Pytest 测试 ---
def test_get_feature_names_actual():
    """
    测试 get_static_feature_names 方法在不mocking的情况下, 是否返回预期的字段名列表.
    """
    # 定义期望的字段名列表
    # 顺序应该与数据类定义中的顺序一致 (排除 raw_column_name)
    expected_feature_names = [
        "battery_id",
        "charge_rate",
        "discharge_rate",
        "temperature",
        "pressure",
        "dod",
    ]

    # 调用被测试的类方法
    actual_feature_names = BatteryStaticFeatures.get_static_feature_names()

    # 断言返回的列表与期望的列表完全一致
    assert actual_feature_names == expected_feature_names, "获取的特征名称列表与期望不符或顺序错误."


def test_get_feature_names_return_type():
    """
    测试 get_static_feature_names 方法返回的类型是否为列表.
    """
    feature_names = BatteryStaticFeatures.get_static_feature_names()
    # 断言返回值的类型是 list
    assert isinstance(feature_names, list), "返回类型不是列表."


def test_get_feature_names_with_mock(mocker):
    """
    测试 get_static_feature_names 方法在mocking dataclasses.fields的情况下, 是否正确过滤并返回字段名列表.
    """
    # 创建 Mock 字段列表
    mock_fields_list = [
        mocker.Mock(**{"name": fname})
        for fname in [
            "battery_id",
            "charge_rate",
            "discharge_rate",
            "temperature",
            "pressure",
            "dod",
            "raw_column_name",
        ]
    ]

    with mocker.patch.object(dataclasses, "fields", return_value=mock_fields_list):
        # 调用方法
        actual_feature_names = BatteryStaticFeatures.get_static_feature_names()
        # 预期结果
        expected_feature_names = [
            "battery_id",
            "charge_rate",
            "discharge_rate",
            "temperature",
            "pressure",
            "dod",
        ]
        # 断言
        assert (
            actual_feature_names == expected_feature_names
        ), f"预期 {expected_feature_names}, 但得到 {actual_feature_names}"
