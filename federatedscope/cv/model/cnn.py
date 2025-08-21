"""计算机视觉卷积神经网络模型模块

该模块实现了用于联邦学习的各种卷积神经网络架构，包括轻量级 CNN 和 VGG 网络。
这些模型专门为联邦学习场景设计，适用于图像分类任务。

主要模型：
- ConvNet2: 2层卷积网络，适用于简单图像分类任务
- ConvNet5: 5层卷积网络，具有更强的特征提取能力
- VGG11: 基于 VGG 架构的11层网络，适用于复杂图像分类

特性：
- 支持批量归一化 (Batch Normalization)
- 支持 Dropout 正则化
- 灵活的输入尺寸配置
- 可配置的隐藏层大小和类别数

适用数据集：
- CIFAR-10/100
- FEMNIST
- CelebA
- 其他图像分类数据集
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Conv2d, BatchNorm2d
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU


class ConvNet2(Module):
    """2层卷积神经网络
    
    一个轻量级的卷积神经网络，包含2个卷积层和2个全连接层。
    适用于简单的图像分类任务，如 CIFAR-10、FEMNIST 等。
    
    网络结构：
    - Conv2d(in_channels, 32, 5) + BN + ReLU + MaxPool2d(2)
    - Conv2d(32, 64, 5) + BN + ReLU + MaxPool2d(2)
    - Flatten + Dropout
    - Linear(hidden_size, hidden) + ReLU + Dropout
    - Linear(hidden, class_num)
    
    Args:
        in_channels (int): 输入图像的通道数（如RGB图像为3，灰度图像为1）
        h (int): 输入图像的高度，默认为32
        w (int): 输入图像的宽度，默认为32
        hidden (int): 第一个全连接层的隐藏单元数，默认为2048
        class_num (int): 分类类别数，默认为10
        use_bn (bool): 是否使用批量归一化，默认为True
        dropout (float): Dropout概率，默认为0.0
    
    Note:
        - 输入图像尺寸会通过两次MaxPool2d(2)缩小到原来的1/4
        - 使用ReLU激活函数和可选的批量归一化
        - 支持Dropout正则化防止过拟合
    """
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 use_bn=True,
                 dropout=.0):
        super(ConvNet2, self).__init__()

        # 卷积层定义
        self.conv1 = Conv2d(in_channels, 32, 5, padding=2)  # 第一个卷积层：输入->32通道
        self.conv2 = Conv2d(32, 64, 5, padding=2)           # 第二个卷积层：32->64通道
        
        # 批量归一化层（可选）
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = BatchNorm2d(32)  # 第一个卷积层的BN
            self.bn2 = BatchNorm2d(64)  # 第二个卷积层的BN

        # 全连接层定义
        # 计算经过两次MaxPool2d(2)后的特征图尺寸
        self.fc1 = Linear((h // 2 // 2) * (w // 2 // 2) * 64, hidden)
        self.fc2 = Linear(hidden, class_num)

        # 激活函数和其他层
        self.relu = ReLU(inplace=True)  # ReLU激活函数
        self.maxpool = MaxPool2d(2)     # 2x2最大池化
        self.dropout = dropout          # Dropout概率

    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入图像张量，形状为 (batch_size, in_channels, h, w)
        
        Returns:
            torch.Tensor: 分类logits，形状为 (batch_size, class_num)
        """
        # 第一个卷积块：Conv -> BN(可选) -> ReLU -> MaxPool
        x = self.bn1(self.conv1(x)) if self.use_bn else self.conv1(x)
        x = self.maxpool(self.relu(x))
        
        # 第二个卷积块：Conv -> BN(可选) -> ReLU -> MaxPool
        x = self.bn2(self.conv2(x)) if self.use_bn else self.conv2(x)
        x = self.maxpool(self.relu(x))
        
        # 展平特征图并通过全连接层
        x = Flatten()(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


class ConvNet5(Module):
    """5层卷积神经网络
    
    一个中等复杂度的卷积神经网络，包含5个卷积层和2个全连接层。
    相比ConvNet2具有更强的特征提取能力，适用于更复杂的图像分类任务。
    
    网络结构：
    - Conv2d(in_channels, 32, 5) + BN + ReLU + MaxPool2d(2)
    - Conv2d(32, 64, 5) + BN + ReLU + MaxPool2d(2)
    - Conv2d(64, 64, 5) + BN + ReLU + MaxPool2d(2)
    - Conv2d(64, 128, 5) + BN + ReLU + MaxPool2d(2)
    - Conv2d(128, 128, 5) + BN + ReLU + MaxPool2d(2)
    - Flatten + Dropout
    - Linear(hidden_size, hidden) + ReLU + Dropout
    - Linear(hidden, class_num)
    
    Args:
        in_channels (int): 输入图像的通道数
        h (int): 输入图像的高度，默认为32
        w (int): 输入图像的宽度，默认为32
        hidden (int): 第一个全连接层的隐藏单元数，默认为2048
        class_num (int): 分类类别数，默认为10
        dropout (float): Dropout概率，默认为0.0
    
    Note:
        - 输入图像尺寸会通过五次MaxPool2d(2)缩小到原来的1/32
        - 所有卷积层都使用批量归一化
        - 通道数逐渐增加：32 -> 64 -> 64 -> 128 -> 128
    """
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 dropout=.0):
        super(ConvNet5, self).__init__()

        # 第一个卷积块：输入 -> 32通道
        self.conv1 = Conv2d(in_channels, 32, 5, padding=2)
        self.bn1 = BatchNorm2d(32)

        # 第二个卷积块：32 -> 64通道
        self.conv2 = Conv2d(32, 64, 5, padding=2)
        self.bn2 = BatchNorm2d(64)

        # 第三个卷积块：64 -> 64通道
        self.conv3 = Conv2d(64, 64, 5, padding=2)
        self.bn3 = BatchNorm2d(64)

        # 第四个卷积块：64 -> 128通道
        self.conv4 = Conv2d(64, 128, 5, padding=2)
        self.bn4 = BatchNorm2d(128)

        # 第五个卷积块：128 -> 128通道
        self.conv5 = Conv2d(128, 128, 5, padding=2)
        self.bn5 = BatchNorm2d(128)

        # 激活函数和池化层
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)

        # 全连接层（经过5次池化后的特征图尺寸）
        self.fc1 = Linear(
            (h // 2 // 2 // 2 // 2 // 2) * (w // 2 // 2 // 2 // 2 // 2) * 128,
            hidden)
        self.fc2 = Linear(hidden, class_num)

        self.dropout = dropout

    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入图像张量，形状为 (batch_size, in_channels, h, w)
        
        Returns:
            torch.Tensor: 分类logits，形状为 (batch_size, class_num)
        """
        # 第一个卷积块
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # 第二个卷积块
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        # 第三个卷积块
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        # 第四个卷积块
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)

        # 第五个卷积块
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxpool(x)

        # 展平并通过全连接层
        x = Flatten()(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


class VGG11(Module):
    """VGG11 卷积神经网络
    
    基于VGG架构的11层卷积神经网络，是VGG系列中较轻量的版本。
    采用小卷积核(3x3)和深层网络结构，在保持较少参数的同时获得良好性能。
    
    网络结构：
    - Conv2d(in_channels, 64, 3) + BN + ReLU + MaxPool2d(2)
    - Conv2d(64, 128, 3) + BN + ReLU + MaxPool2d(2)
    - Conv2d(128, 256, 3) + BN + ReLU + MaxPool2d(2)
    - Conv2d(256, 256, 3) + BN + ReLU + MaxPool2d(2)
    - Conv2d(256, 512, 3) + BN + ReLU + MaxPool2d(2)
    - Conv2d(512, 512, 3) + BN + ReLU + MaxPool2d(2)
    - Conv2d(512, 512, 3) + BN + ReLU + MaxPool2d(2)
    - Conv2d(512, 512, 3) + BN + ReLU + MaxPool2d(2)
    - Flatten + Dropout
    - Linear(hidden_size, hidden) + ReLU + Dropout
    - Linear(hidden, hidden) + ReLU + Dropout
    - Linear(hidden, class_num)
    
    Args:
        in_channels (int): 输入图像的通道数
        h (int): 输入图像的高度，默认为32
        w (int): 输入图像的宽度，默认为32
        hidden (int): 全连接层的隐藏单元数，默认为128
        class_num (int): 分类类别数，默认为10
        dropout (float): Dropout概率，默认为0.0
    
    Note:
        - 使用3x3小卷积核，增加网络深度
        - 通道数逐渐增加：64 -> 128 -> 256 -> 512
        - 所有卷积层都使用批量归一化
        - 适用于复杂的图像分类任务
    """
    def __init__(self,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=128,
                 class_num=10,
                 dropout=.0):
        super(VGG11, self).__init__()

        # 第一个卷积块：输入 -> 64通道
        self.conv1 = Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = BatchNorm2d(64)

        # 第二个卷积块：64 -> 128通道
        self.conv2 = Conv2d(64, 128, 3, padding=1)
        self.bn2 = BatchNorm2d(128)

        # 第三个卷积块：128 -> 256通道
        self.conv3 = Conv2d(128, 256, 3, padding=1)
        self.bn3 = BatchNorm2d(256)

        # 第四个卷积块：256 -> 256通道
        self.conv4 = Conv2d(256, 256, 3, padding=1)
        self.bn4 = BatchNorm2d(256)

        # 第五个卷积块：256 -> 512通道
        self.conv5 = Conv2d(256, 512, 3, padding=1)
        self.bn5 = BatchNorm2d(512)

        # 第六个卷积块：512 -> 512通道
        self.conv6 = Conv2d(512, 512, 3, padding=1)
        self.bn6 = BatchNorm2d(512)

        # 第七个卷积块：512 -> 512通道
        self.conv7 = Conv2d(512, 512, 3, padding=1)
        self.bn7 = BatchNorm2d(512)

        # 第八个卷积块：512 -> 512通道
        self.conv8 = Conv2d(512, 512, 3, padding=1)
        self.bn8 = BatchNorm2d(512)

        # 激活函数和池化层
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)

        # 全连接层（经过5次池化后的特征图尺寸）
        self.fc1 = Linear(
            (h // 2 // 2 // 2 // 2 // 2) * (w // 2 // 2 // 2 // 2 // 2) * 512,
            hidden)
        self.fc2 = Linear(hidden, hidden)
        self.fc3 = Linear(hidden, class_num)

        self.dropout = dropout

    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入图像张量，形状为 (batch_size, in_channels, h, w)
        
        Returns:
            torch.Tensor: 分类logits，形状为 (batch_size, class_num)
        """
        # 第一个卷积块 + 池化
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # 第二个卷积块 + 池化
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        # 第三个卷积块 + 池化
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        # 第四个卷积块 + 池化
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)

        # 第五个卷积块 + 池化
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxpool(x)

        # 第六个卷积块 + 池化
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.maxpool(x)

        # 第七个卷积块 + 池化
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.maxpool(x)

        # 第八个卷积块 + 池化
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.maxpool(x)

        # 展平并通过全连接层
        x = Flatten()(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x
