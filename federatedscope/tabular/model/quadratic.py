"""二次模型

该模块提供了用于表格数据的二次模型实现。
主要功能包括：
- 实现二次函数形式的神经网络模型
- 适用于表格数据的回归和分类任务
- 支持联邦学习环境下的参数优化

模型特性：
- 基于二次函数的参数化模型
- 可学习的参数向量
- 支持矩阵运算的高效计算

应用场景：
- 表格数据分析
- 二次优化问题
- 联邦学习中的简单模型测试
"""

import torch


class QuadraticModel(torch.nn.Module):
    """二次模型类
    
    实现基于二次函数的神经网络模型，适用于表格数据处理。
    模型计算形式为：f(A) = x^T * A * x，其中x是可学习参数。
    
    参数:
        in_channels (int): 输入特征的维度
        class_num (int): 类别数量（当前实现中未直接使用）
    
    属性:
        x (torch.nn.Parameter): 可学习的参数向量，形状为 (in_channels, 1)
    
    注意:
        - 参数x初始化为[-10.0, 10.0]范围内的均匀分布
        - 模型输出为标量值，适用于回归任务
    """
    def __init__(self, in_channels, class_num):
        """初始化二次模型
        
        参数:
            in_channels (int): 输入特征的维度
            class_num (int): 类别数量（保留参数，当前实现中未使用）
        """
        super(QuadraticModel, self).__init__()
        # 创建参数向量，形状为 (in_channels, 1)
        x = torch.ones((in_channels, 1))
        # 将参数初始化为[-10.0, 10.0]范围内的均匀分布，并设置为可学习参数
        self.x = torch.nn.parameter.Parameter(x.uniform_(-10.0, 10.0).float())

    def forward(self, A):
        """前向传播
        
        计算二次函数：f(A) = x^T * A * x
        
        参数:
            A (torch.Tensor): 输入矩阵，形状为 (batch_size, in_channels, in_channels)
        
        返回:
            torch.Tensor: 输出结果，形状为 (batch_size,)
        
        注意:
            - 计算过程：先计算 A * x，然后计算 x^T * (A * x)
            - 最后对最后一个维度求和得到标量输出
        """
        # 计算二次函数：x^T * A * x
        # 1. 计算 A * x
        # 2. 计算 x * (A * x) 得到逐元素乘积
        # 3. 对最后一个维度求和
        return torch.sum(self.x * torch.matmul(A, self.x), -1)
