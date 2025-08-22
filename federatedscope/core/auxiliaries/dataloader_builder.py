"""数据加载器构建器模块

该模块负责根据配置创建不同类型的数据加载器实例。
支持多种数据加载器类型，适用于不同的机器学习任务和数据格式。

支持的数据加载器类型：
- base：标准PyTorch数据加载器
- raw：无数据加载器，直接返回数据集
- pyg：PyTorch Geometric数据加载器（用于图数据）
- graphsaint-rw：GraphSAINT随机游走采样器
- neighbor：邻居采样器（用于大图训练）
- mf：矩阵分解数据加载器

主要功能：
1. 根据配置类型选择合适的数据加载器
2. 处理训练和测试阶段的不同配置
3. 支持图神经网络的特殊采样策略
4. 处理大语言模型的数据整理
5. 自动过滤和适配参数
"""

from federatedscope.core.data.utils import filter_dict

# 尝试导入PyTorch相关模块
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


def get_dataloader(dataset, config, split='train'):
    """根据配置创建数据加载器实例
    
    该函数根据指定的配置创建相应类型的数据加载器，用于批量加载和处理数据。
    不同的数据加载器适用于不同的机器学习任务和数据格式。

    参数:
        dataset: 待加载的数据集对象
        config: 包含batch_size、shuffle等配置的对象
        split (str): 当前数据分割（默认为'train'）
                    如果split为'test'，cfg.dataloader.shuffle将为False
                    在PyG中，'test'分割默认使用NeighborSampler

    返回:
        dataloader: 根据配置创建的数据加载器实例
        
    注意:
        数据加载器类型与对应类的映射关系：
        ========================  ===============================
        数据加载器类型              来源
        ========================  ===============================
        ``raw``                   无数据加载器
        ``base``                  ``torch.utils.data.DataLoader``
        ``pyg``                   ``torch_geometric.loader.DataLoader``
        ``graphsaint-rw``         \
        ``torch_geometric.loader.GraphSAINTRandomWalkSampler``
        ``neighbor``              ``torch_geometric.loader.NeighborSampler``
        ``mf``                    ``federatedscope.mf.dataloader.MFDataLoader``
        ========================  ===============================
    """
    # 数据加载器构建器目前仅支持PyTorch后端
    if config.backend != 'torch':
        return None

    # 根据配置类型选择相应的数据加载器类
    if config.dataloader.type == 'base':
        # 标准PyTorch数据加载器
        from torch.utils.data import DataLoader
        loader_cls = DataLoader
    elif config.dataloader.type == 'raw':
        # 无数据加载器，直接返回数据集
        return dataset
    elif config.dataloader.type == 'pyg':
        # PyTorch Geometric数据加载器
        loader_cls = PyGDataLoader
    elif config.dataloader.type == 'graphsaint-rw':
        # GraphSAINT随机游走采样器
        if split == 'train':
            from torch_geometric.loader import GraphSAINTRandomWalkSampler
            loader_cls = GraphSAINTRandomWalkSampler
        else:
            # 测试时使用邻居采样器
            from torch_geometric.loader import NeighborSampler
            loader_cls = NeighborSampler
    elif config.dataloader.type == 'neighbor':
        # 邻居采样器（用于大图训练）
        from torch_geometric.loader import NeighborSampler
        loader_cls = NeighborSampler
    elif config.dataloader.type == 'mf':
        # 矩阵分解数据加载器
        from federatedscope.mf.dataloader import MFDataLoader
        loader_cls = MFDataLoader
    else:
        raise ValueError(f'data.loader.type {config.data.loader.type} '
                         f'not found!')

    # 获取数据加载器配置参数
    raw_args = dict(config.dataloader)
    
    # 根据数据分割调整参数
    if split != 'train':
        # 测试阶段不打乱数据
        raw_args['shuffle'] = False
        raw_args['sizes'] = [-1]  # 使用所有邻居
        raw_args['drop_last'] = False
        # 图联邦学习评估的特殊处理
        if config.dataloader.type in ['graphsaint-rw', 'neighbor']:
            raw_args['batch_size'] = 4096
            dataset = dataset[0].edge_index
    else:
        # 训练阶段的数据集处理
        if config.dataloader.type in ['graphsaint-rw']:
            # 使用原始图
            dataset = dataset[0]
        elif config.dataloader.type in ['neighbor']:
            # 使用原始图的边索引
            dataset = dataset[0].edge_index
    
    # 过滤出数据加载器类支持的参数
    filtered_args = filter_dict(loader_cls.__init__, raw_args)

    # 处理大语言模型数据的特殊情况
    if config.data.type.lower().endswith('@llm'):
        from federatedscope.llm.dataloader import get_tokenizer, \
            LLMDataCollator
        # 解析模型名称和模型中心
        model_name, model_hub = config.model.type.split('@')
        # 获取分词器
        tokenizer, _ = get_tokenizer(model_name, config.data.root,
                                     config.llm.tok_len, model_hub)
        # 创建LLM数据整理器
        data_collator = LLMDataCollator(tokenizer=tokenizer)
        filtered_args['collate_fn'] = data_collator

    # 创建并返回数据加载器实例
    dataloader = loader_cls(dataset, **filtered_args)
    return dataloader
