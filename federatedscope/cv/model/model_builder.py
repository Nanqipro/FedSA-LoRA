"""计算机视觉模型构建器

该模块提供了用于构建计算机视觉模型的工厂函数。
支持多种卷积神经网络架构，包括ConvNet2、ConvNet5和VGG11。

主要功能：
- 根据配置参数动态创建CNN模型
- 支持不同复杂度的网络架构
- 统一的模型创建接口

支持的模型：
- convnet2: 2层卷积神经网络，适用于简单任务
- convnet5: 5层卷积神经网络，适用于中等复杂度任务
- vgg11: VGG11网络，适用于复杂的图像分类任务

Author: FederatedScope团队
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from federatedscope.cv.model.cnn import ConvNet2, ConvNet5, VGG11


def get_cnn(model_config, input_shape):
    """获取卷积神经网络模型
    
    根据模型配置和输入形状创建相应的CNN模型。
    支持ConvNet2、ConvNet5和VGG11三种网络架构。
    
    Args:
        model_config: 模型配置对象，包含以下属性：
            - type (str): 模型类型，可选值：'convnet2', 'convnet5', 'vgg11'
            - hidden (int): 隐藏层单元数
            - out_channels (int): 输出类别数
            - dropout (float): Dropout概率
        input_shape (tuple): 输入数据形状
            - 格式：(batch_size, in_channels, h, w) 或 (in_channels, h, w)
            - in_channels: 输入通道数
            - h: 图像高度
            - w: 图像宽度
    
    Returns:
        torch.nn.Module: 创建的CNN模型实例
    
    Raises:
        ValueError: 当指定的模型类型不支持时抛出异常
    
    Example:
        >>> from types import SimpleNamespace
        >>> config = SimpleNamespace(type='convnet2', hidden=128, out_channels=10, dropout=0.1)
        >>> model = get_cnn(config, (3, 32, 32))
        >>> print(type(model).__name__)  # ConvNet2
    
    Note:
        - 输入形状的最后三个维度分别对应通道数、高度和宽度
        - 所有模型都支持批量归一化和Dropout
        - 模型类型不区分大小写
    """
    # 检查任务类型并根据输入形状创建模型
    # input_shape: (batch_size, in_channels, h, w) 或 (in_channels, h, w)
    if model_config.type == 'convnet2':
        # 创建2层卷积网络，适用于简单的图像分类任务
        model = ConvNet2(in_channels=input_shape[-3],
                         h=input_shape[-2],
                         w=input_shape[-1],
                         hidden=model_config.hidden,
                         class_num=model_config.out_channels,
                         dropout=model_config.dropout)
    elif model_config.type == 'convnet5':
        # 创建5层卷积网络，适用于中等复杂度的图像分类任务
        model = ConvNet5(in_channels=input_shape[-3],
                         h=input_shape[-2],
                         w=input_shape[-1],
                         hidden=model_config.hidden,
                         class_num=model_config.out_channels,
                         dropout=model_config.dropout)
    elif model_config.type == 'vgg11':
        # 创建VGG11网络，适用于复杂的图像分类任务
        model = VGG11(in_channels=input_shape[-3],
                      h=input_shape[-2],
                      w=input_shape[-1],
                      hidden=model_config.hidden,
                      class_num=model_config.out_channels,
                      dropout=model_config.dropout)
    else:
        # 抛出异常，提示不支持的模型类型
        raise ValueError(f'No model named {model_config.type}!')

    return model
