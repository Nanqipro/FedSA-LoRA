"""FederatedScope 模块注册系统

该文件实现了 FederatedScope 框架的模块注册机制，允许用户动态注册和扩展各种组件：
- 数据加载器 (Data Loaders)
- 模型 (Models)
- 训练器 (Trainers)
- 配置 (Configs)
- 指标 (Metrics)
- 损失函数 (Criterions)
- 正则化器 (Regularizers)
- 数据变换 (Transforms)
- 数据分割器 (Splitters)
- 调度器 (Schedulers)
- 优化器 (Optimizers)
- 工作器 (Workers)

通过注册机制，用户可以轻松扩展框架功能，添加自定义组件。
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging

logger = logging.getLogger(__name__)


def register(key, module, module_dict):
    """通用注册函数
    
    Args:
        key (str): 模块的注册键名
        module: 要注册的模块或类
        module_dict (dict): 存储注册模块的字典
    """
    if key in module_dict:
        logger.warning(
            'Key {} is already pre-defined, overwritten.'.format(key))
    module_dict[key] = module


# 数据加载器注册字典
data_dict = {}


def register_data(key, module):
    """注册数据加载器
    
    Args:
        key (str): 数据加载器的注册键名
        module: 数据加载器类或函数
    """
    register(key, module, data_dict)


# 模型注册字典
model_dict = {}


def register_model(key, module):
    """注册模型
    
    Args:
        key (str): 模型的注册键名
        module: 模型类
    """
    register(key, module, model_dict)


# 训练器注册字典
trainer_dict = {}


def register_trainer(key, module):
    """注册训练器
    
    Args:
        key (str): 训练器的注册键名
        module: 训练器类
    """
    register(key, module, trainer_dict)


# 配置注册字典
config_dict = {}


def register_config(key, module):
    """注册配置
    
    Args:
        key (str): 配置的注册键名
        module: 配置类或函数
    """
    register(key, module, config_dict)


# 指标注册字典
metric_dict = {}


def register_metric(key, module):
    """注册评估指标
    
    Args:
        key (str): 指标的注册键名
        module: 指标计算函数
    """
    register(key, module, metric_dict)


# 损失函数注册字典
criterion_dict = {}


def register_criterion(key, module):
    """注册损失函数
    
    Args:
        key (str): 损失函数的注册键名
        module: 损失函数类
    """
    register(key, module, criterion_dict)


# 正则化器注册字典
regularizer_dict = {}


def register_regularizer(key, module):
    """注册正则化器
    
    Args:
        key (str): 正则化器的注册键名
        module: 正则化器类或函数
    """
    register(key, module, regularizer_dict)


# 辅助数据加载器（隐私影响分析）注册字典
auxiliary_data_loader_PIA_dict = {}


def register_auxiliary_data_loader_PIA(key, module):
    """注册隐私影响分析辅助数据加载器
    
    Args:
        key (str): 辅助数据加载器的注册键名
        module: 辅助数据加载器类或函数
    """
    register(key, module, auxiliary_data_loader_PIA_dict)


# 数据变换注册字典
transform_dict = {}


def register_transform(key, module):
    """注册数据变换
    
    Args:
        key (str): 数据变换的注册键名
        module: 数据变换类或函数
    """
    register(key, module, transform_dict)


# 数据分割器注册字典
splitter_dict = {}


def register_splitter(key, module):
    """注册数据分割器
    
    Args:
        key (str): 数据分割器的注册键名
        module: 数据分割器类或函数
    """
    register(key, module, splitter_dict)


# 学习率调度器注册字典
scheduler_dict = {}


def register_scheduler(key, module):
    """注册学习率调度器
    
    Args:
        key (str): 调度器的注册键名
        module: 调度器类
    """
    register(key, module, scheduler_dict)


# 优化器注册字典
optimizer_dict = {}


def register_optimizer(key, module):
    """注册优化器
    
    Args:
        key (str): 优化器的注册键名
        module: 优化器类
    """
    register(key, module, optimizer_dict)


# 工作器注册字典
worker_dict = {}


def register_worker(key, module):
    """注册工作器
    
    Args:
        key (str): 工作器的注册键名
        module: 工作器类
    """
    register(key, module, worker_dict)