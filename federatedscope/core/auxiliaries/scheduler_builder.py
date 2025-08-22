"""学习率调度器构建器模块

该模块负责根据调度器类型和参数创建相应的学习率调度器。
支持多种调度策略，用于在训练过程中动态调整学习率。

支持的调度器类型：
- PyTorch内置调度器：StepLR, ExponentialLR, CosineAnnealingLR等
- 自定义调度器：warmup_step, warmup_noam
- 注册的扩展调度器

主要功能：
1. 根据调度器类型和优化器创建调度器实例
2. 支持warmup预热策略
3. 处理配置参数的清理和验证
4. 支持注册机制的自定义调度器
5. 提供PyTorch内置调度器的统一接口
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

# 尝试导入贡献的调度器模块
try:
    from federatedscope.contrib.scheduler import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.scheduler`, some modules are not '
        f'available.')


def get_scheduler(optimizer, type, **kwargs):
    """创建学习率调度器实例
    
    该函数根据调度器类型和优化器创建相应的学习率调度器实例。
    支持多种调度策略，包括预热、余弦退火等。

    参数:
        optimizer: 需要调度的优化器
        type (str): 调度器类型名称
        **kwargs: 调度器的额外参数

    返回:
        调度器实例，如果类型为空或PyTorch不可用则返回None

    注意:
        请参考 ``contrib.scheduler.example`` 来实现自定义调度器。
        
    支持的调度器类型:
        - warmup_step: 带预热的线性衰减调度器
        - warmup_noam: Noam预热调度器（Transformer论文中使用）
        - PyTorch内置调度器: StepLR, ExponentialLR, CosineAnnealingLR等
    """
    # 防止用户未调用cfg.freeze()，深拷贝参数
    tmp_kwargs = copy.deepcopy(kwargs)
    
    # 清理配置中的元信息字段
    if '__help_info__' in tmp_kwargs:
        del tmp_kwargs['__help_info__']
    if '__cfg_check_funcs__' in tmp_kwargs:
        del tmp_kwargs['__cfg_check_funcs__']
    if 'is_ready_for_run' in tmp_kwargs:
        del tmp_kwargs['is_ready_for_run']
    if 'warmup_ratio' in tmp_kwargs:
        del tmp_kwargs['warmup_ratio']
    
    # 提取预热相关参数
    if 'warmup_steps' in tmp_kwargs:
        warmup_steps = tmp_kwargs['warmup_steps']
        del tmp_kwargs['warmup_steps']
    if 'total_steps' in tmp_kwargs:
        total_steps = tmp_kwargs['total_steps']
        del tmp_kwargs['total_steps']

    # 尝试使用注册的调度器函数
    for func in register.scheduler_dict.values():
        scheduler = func(optimizer, type, **tmp_kwargs)
        if scheduler is not None:
            return scheduler

    # 处理自定义的预热调度器
    if type == 'warmup_step':
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(cur_step):
            # 预热阶段：线性增长
            if cur_step < warmup_steps:
                return float(cur_step) / float(max(1, warmup_steps))
            # 衰减阶段：线性衰减
            return max(
                0.0,
                float(total_steps - cur_step) /
                float(max(1, total_steps - warmup_steps)))

        return LambdaLR(optimizer, lr_lambda)
    elif type == 'warmup_noam':
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(cur_step):
            # Noam调度器：Transformer论文中使用的调度策略
            return min(
                max(1, cur_step)**(-0.5),
                max(1, cur_step) * warmup_steps**(-1.5))

        return LambdaLR(optimizer, lr_lambda)

    # 如果PyTorch不可用或类型为空，返回None
    if torch is None or type == '':
        return None
    
    # 处理字符串类型的调度器名称
    if isinstance(type, str):
        # 检查是否为PyTorch内置调度器
        if hasattr(torch.optim.lr_scheduler, type):
            return getattr(torch.optim.lr_scheduler, type)(optimizer,
                                                           **tmp_kwargs)
        else:
            raise NotImplementedError(
                'Scheduler {} not implement'.format(type))
    else:
        raise TypeError('Scheduler type must be a string')
