"""优化器构建器模块

该模块负责根据配置创建和初始化各种优化器，支持PyTorch内置的所有优化器类型。

支持的优化器类型：
- SGD：随机梯度下降
- Adam：自适应矩估计
- AdamW：带权重衰减的Adam
- RMSprop：均方根传播
- Adagrad：自适应梯度算法
- Adadelta：自适应学习率方法
- Adamax：基于无穷范数的Adam变体
- ASGD：平均随机梯度下降
- LBFGS：有限内存BFGS
- Rprop：弹性反向传播
- SparseAdam：稀疏张量的Adam

主要功能：
1. 根据模型和配置创建优化器实例
2. 处理配置参数的清理和验证
3. 支持注册的自定义优化器
4. 兼容PyTorch的所有内置优化器
"""

import copy
import logging
import federatedscope.register as register

logger = logging.getLogger(__name__)

# 尝试导入PyTorch
try:
    import torch
except ImportError:
    torch = None

# 尝试导入贡献的优化器模块
try:
    from federatedscope.contrib.optimizer import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.optimizer`, some modules are not '
        f'available.')


def get_optimizer(model, type, lr, **kwargs):
    """创建用于优化模型的优化器实例
    
    该函数根据指定的优化器类型、学习率和其他参数创建优化器实例。
    支持PyTorch内置的所有优化器类型以及注册的自定义优化器。

    参数:
        model: 待优化的模型对象
        type (str): 优化器类型，参见PyTorch文档
                   https://pytorch.org/docs/stable/optim.html
        lr (float): 学习率
        **kwargs: 其他优化器参数的字典

    返回:
        optimizer: 实例化的优化器对象
        
    注意:
        如果PyTorch未安装，将返回None
    """
    # 如果PyTorch未安装，返回None
    if torch is None:
        return None
        
    # 清理配置参数，防止用户未调用cfg.freeze()的情况
    tmp_kwargs = copy.deepcopy(kwargs)
    if '__help_info__' in tmp_kwargs:
        del tmp_kwargs['__help_info__']
    if '__cfg_check_funcs__' in tmp_kwargs:
        del tmp_kwargs['__cfg_check_funcs__']
    if 'is_ready_for_run' in tmp_kwargs:
        del tmp_kwargs['is_ready_for_run']

    # 尝试使用注册的优化器函数创建优化器
    for func in register.optimizer_dict.values():
        optimizer = func(model, type, lr, **tmp_kwargs)
        if optimizer is not None:
            return optimizer

    # 使用PyTorch内置优化器
    if isinstance(type, str):
        if hasattr(torch.optim, type):
            # 根据模型类型选择参数传递方式
            if isinstance(model, torch.nn.Module):
                # 对于PyTorch模型，传递模型参数
                return getattr(torch.optim, type)(model.parameters(), lr,
                                                  **tmp_kwargs)
            else:
                # 对于其他类型，直接传递模型对象
                return getattr(torch.optim, type)(model, lr, **tmp_kwargs)
        else:
            raise NotImplementedError(
                'Optimizer {} not implement'.format(type))
    else:
        raise TypeError('Optimizer type must be a string')
