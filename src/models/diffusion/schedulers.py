from diffusers.schedulers import (
    DDIMScheduler,
    DDPMParallelScheduler,
    DDPMScheduler,
    EulerDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

# 使用字典来映射配置文件中的字符串名称到 Diffusers 的调度器类
SCHEDULERS_MAP = {
    "ddpm": DDPMScheduler,
    "ddim": DDIMScheduler,
    "euler": EulerDiscreteScheduler,
    "pndm": PNDMScheduler,
    "lms": LMSDiscreteScheduler,
    "kdpm2": KDPM2DiscreteScheduler,
    "ddpm_parallel": DDPMParallelScheduler,
}


def get_scheduler(config: dict):
    """
    根据配置字典实例化并返回 Hugging Face Diffusers 调度器。

    这个函数提供了一个统一的接口来获取不同类型的调度器。

    Args:
        config: 包含调度器配置的字典，通常从 YAML 文件加载。
                需要包含 'scheduler_type' 字段，以及该调度器类初始化所需的其他参数。
                例如:
                {
                  "scheduler_type": "ddpm", # 调度器类型 (字符串，必须在 SCHEDULERS_MAP 中)
                  "timesteps": 1000,      # 总的时间步 T
                  "beta_schedule": "linear", # beta 调度方式 (如 "linear", "cosine")
                  "prediction_type": "epsilon", # 模型预测的类型 (如 "epsilon", "v_prediction", "sample")
                  ... # 其他特定调度器需要的参数，请参考 Diffusers 官方文档
                }

    Returns:
        实例化后的 Diffusers 调度器对象。

    Raises:
        ValueError: 如果 config 格式不正确或调度器类型不支持。
        TypeError: 如果调度器参数与所选调度器类不匹配。
    """
    if not isinstance(config, dict):
        raise ValueError("调度器配置必须是一个字典。")

    scheduler_type = config.get("scheduler_type")
    if scheduler_type is None:
        raise ValueError("调度器配置中必须包含 'scheduler_type' 字段。")

    # 检查调度器类型是否支持
    if scheduler_type not in SCHEDULERS_MAP:
        raise ValueError(
            f"不支持的调度器类型: '{scheduler_type}'. "
            f"当前支持的类型有: {list(SCHEDULERS_MAP.keys())}"
        )

    # 获取对应的调度器类
    scheduler_class = SCHEDULERS_MAP[scheduler_type]

    # 准备用于实例化调度器的参数
    # 从 config 中过滤掉 'scheduler_type'，因为它不是调度器类的初始化参数
    scheduler_params = {k: v for k, v in config.items() if k != "scheduler_type"}

    # 实例化调度器
    try:
        # 使用 **scheduler_params 将字典中的参数解包传递给类构造函数
        scheduler = scheduler_class(**scheduler_params)
        print(f"成功实例化 {scheduler_type} 调度器，参数: {scheduler_params}")
        return scheduler
    except TypeError:
        # 捕获参数类型或缺失错误，提供更友好的提示
        print(f"错误: 初始化调度器 '{scheduler_type}' 时参数不匹配或缺失。")
        print(f"使用的参数: {scheduler_params}")
        print(f"请检查 '{scheduler_type}' 类在其初始化函数中需要的准确参数。")
        raise
    except Exception:
        # 捕获其他可能的实例化错误
        print(f"初始化调度器 '{scheduler_type}' 时发生未知错误，参数: {scheduler_params}")
        raise
