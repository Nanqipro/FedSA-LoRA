"""图神经网络模型构建器

该模块提供了用于构建图神经网络(GNN)模型的工厂函数。
支持多种GNN架构，适用于节点级、边级和图级任务。

主要功能：
- 根据配置参数动态创建GNN模型
- 支持多种图神经网络架构
- 适配不同类型的图学习任务
- 统一的模型创建接口

支持的模型：
- GCN: 图卷积网络，适用于节点分类
- SAGE: GraphSAGE，支持大规模图的归纳学习
- GAT: 图注意力网络，使用注意力机制
- GIN: 图同构网络，理论上最强表达能力
- GPR: 图传播网络，适用于异质图
- MPNN: 消息传递神经网络，适用于图级任务

支持的任务：
- node: 节点级任务（节点分类、节点回归）
- link: 边级任务（链接预测、边分类）
- graph: 图级任务（图分类、图回归）

Author: FederatedScope团队
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from federatedscope.gfl.model.gcn import GCN_Net
from federatedscope.gfl.model.sage import SAGE_Net
from federatedscope.gfl.model.gat import GAT_Net
from federatedscope.gfl.model.gin import GIN_Net
from federatedscope.gfl.model.gpr import GPR_Net
from federatedscope.gfl.model.link_level import GNN_Net_Link
from federatedscope.gfl.model.graph_level import GNN_Net_Graph
from federatedscope.gfl.model.mpnn import MPNNs2s


def get_gnn(model_config, input_shape):
    """获取图神经网络模型
    
    根据模型配置和输入形状创建相应的GNN模型。
    支持节点级、边级和图级三种类型的任务。
    
    Args:
        model_config: 模型配置对象，包含以下属性：
            - type (str): 模型类型，可选值：'gcn', 'sage', 'gat', 'gin', 'gpr', 'mpnn'
            - task (str): 任务类型，可选值：'node*', 'link*', 'graph*'
            - out_channels (int): 输出通道数/类别数
            - hidden (int): 隐藏层维度
            - layer (int): 网络层数
            - dropout (float): Dropout概率
            - graph_pooling (str): 图级任务的池化方法（仅图级任务需要）
        input_shape (tuple): 输入数据形状信息
            - x_shape: 节点特征形状
            - num_label: 标签数量
            - num_edge_features: 边特征数量
    
    Returns:
        torch.nn.Module: 创建的GNN模型实例
    
    Raises:
        ValueError: 当指定的模型类型或任务类型不支持时抛出异常
    
    Example:
        >>> from types import SimpleNamespace
        >>> config = SimpleNamespace(type='gcn', task='node_classification', 
        ...                         out_channels=7, hidden=64, layer=2, dropout=0.1)
        >>> model = get_gnn(config, ((2708, 1433), 7, 0))
        >>> print(type(model).__name__)  # GCN_Net
    
    Note:
        - 节点级任务支持：GCN, SAGE, GAT, GIN, GPR
        - 边级任务使用统一的GNN_Net_Link包装器
        - 图级任务支持通用GNN和专门的MPNN
        - 输入形状的解析依赖于具体的数据格式
    """
    # 解析输入形状信息
    x_shape, num_label, num_edge_features = input_shape
    if not num_label:
        num_label = 0
    if model_config.task.startswith('node'):
        # 节点级任务：节点分类、节点回归等
        if model_config.type == 'gcn':
            # 图卷积网络：使用谱域卷积进行节点特征聚合
            # 假设数据是字典格式，键为客户端索引，值为PyG图对象
            model = GCN_Net(x_shape[-1],
                            model_config.out_channels,
                            hidden=model_config.hidden,
                            max_depth=model_config.layer,
                            dropout=model_config.dropout)
        elif model_config.type == 'sage':
            # GraphSAGE：支持大规模图的归纳学习
            model = SAGE_Net(x_shape[-1],
                             model_config.out_channels,
                             hidden=model_config.hidden,
                             max_depth=model_config.layer,
                             dropout=model_config.dropout)
        elif model_config.type == 'gat':
            # 图注意力网络：使用注意力机制聚合邻居信息
            model = GAT_Net(x_shape[-1],
                            model_config.out_channels,
                            hidden=model_config.hidden,
                            max_depth=model_config.layer,
                            dropout=model_config.dropout)
        elif model_config.type == 'gin':
            # 图同构网络：理论上具有最强的表达能力
            model = GIN_Net(x_shape[-1],
                            model_config.out_channels,
                            hidden=model_config.hidden,
                            max_depth=model_config.layer,
                            dropout=model_config.dropout)
        elif model_config.type == 'gpr':
            # 图传播网络：适用于异质图和长距离依赖
            model = GPR_Net(x_shape[-1],
                            model_config.out_channels,
                            hidden=model_config.hidden,
                            K=model_config.layer,
                            dropout=model_config.dropout)
        else:
            # 抛出异常，提示不支持的GNN模型类型
            raise ValueError('not recognized gnn model {}'.format(
                model_config.type))

    elif model_config.task.startswith('link'):
        # 边级任务：链接预测、边分类等
        # 使用统一的GNN_Net_Link包装器，支持多种GNN后端
        model = GNN_Net_Link(x_shape[-1],
                             model_config.out_channels,
                             hidden=model_config.hidden,
                             max_depth=model_config.layer,
                             dropout=model_config.dropout,
                             gnn=model_config.type)
    elif model_config.task.startswith('graph'):
        # 图级任务：图分类、图回归等
        if model_config.type == 'mpnn':
            # 消息传递神经网络：专门为图级任务设计
            # 支持边特征和序列到序列的学习
            model = MPNNs2s(in_channels=x_shape[-1],
                            out_channels=model_config.out_channels,
                            num_nn=num_edge_features,
                            hidden=model_config.hidden)
        else:
            # 通用图级GNN：使用图池化聚合节点信息
            # 支持多种池化策略（mean, max, sum等）
            model = GNN_Net_Graph(x_shape[-1],
                                  max(model_config.out_channels, num_label),
                                  hidden=model_config.hidden,
                                  max_depth=model_config.layer,
                                  dropout=model_config.dropout,
                                  gnn=model_config.type,
                                  pooling=model_config.graph_pooling)
    else:
        # 抛出异常，提示不支持的任务类型
        raise ValueError('not recognized data task {}'.format(
            model_config.task))
    
    return model
