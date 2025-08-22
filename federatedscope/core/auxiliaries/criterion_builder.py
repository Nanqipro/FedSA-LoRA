"""损失函数构建器模块

该模块负责根据损失函数类型创建相应的损失函数实例。
支持PyTorch内置的所有损失函数以及自定义的损失函数。

支持的损失函数类型：
- 分类损失：CrossEntropyLoss, NLLLoss, BCELoss, BCEWithLogitsLoss等
- 回归损失：MSELoss, L1Loss, SmoothL1Loss, HuberLoss等
- 排序损失：MarginRankingLoss, TripletMarginLoss等
- 自然语言处理损失：来自federatedscope.nlp.loss模块
- 对比学习损失：来自federatedscope.cl.loss模块
- 自定义损失：来自federatedscope.contrib.loss模块

主要功能：
1. 根据损失函数类型字符串创建损失函数实例
2. 支持注册的自定义损失函数
3. 兼容PyTorch的所有内置损失函数
4. 处理设备移动（CPU/GPU）
"""

import logging
import federatedscope.register as register

logger = logging.getLogger(__name__)

# 尝试导入PyTorch和相关损失函数模块
try:
    from torch import nn
    from federatedscope.nlp.loss import *      # 自然语言处理损失函数
    from federatedscope.cl.loss import *       # 对比学习损失函数
except ImportError:
    nn = None

# 尝试导入贡献的损失函数模块
try:
    from federatedscope.contrib.loss import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.loss`, some modules are not '
        f'available.')


def get_criterion(criterion_type, device):
    """创建损失函数实例
    
    该函数根据指定的损失函数类型创建相应的损失函数实例。
    支持PyTorch内置的所有损失函数，详见：
    https://pytorch.org/docs/stable/nn.html#loss-functions

    参数:
        criterion_type (str): 损失函数类型名称
        device (str): 设备类型（'cpu'或'gpu'）

    返回:
        criterion: 损失函数实例
        
    注意:
        如果PyTorch未安装，某些损失函数可能不可用
    """
    # 尝试使用注册的损失函数创建实例
    for func in register.criterion_dict.values():
        criterion = func(criterion_type, device)
        if criterion is not None:
            return criterion

    # 使用PyTorch内置损失函数
    if isinstance(criterion_type, str):
        if hasattr(nn, criterion_type):
            # 创建PyTorch内置损失函数实例
            return getattr(nn, criterion_type)()
        else:
            raise NotImplementedError(
                'Criterion {} not implement'.format(criterion_type))
    else:
        raise TypeError('Criterion type must be a string')
