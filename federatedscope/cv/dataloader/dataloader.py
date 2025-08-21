"""计算机视觉数据加载器模块

该模块提供了用于联邦学习的计算机视觉数据集加载功能，主要支持 LEAF 基准数据集。

主要功能：
- 加载 FEMNIST 和 CelebA 数据集
- 数据预处理和变换
- 客户端数据分割
- 数据集格式转换

支持的数据集：
- FEMNIST: 手写字符识别数据集
- CelebA: 名人面部属性数据集
"""

from federatedscope.cv.dataset.leaf_cv import LEAF_CV
from federatedscope.core.auxiliaries.transform_builder import get_transform


def load_cv_dataset(config=None):
    """加载计算机视觉数据集用于联邦学习
    
    该函数支持加载 FEMNIST 和 CelebA 数据集，并将其格式化为适合联邦学习的客户端数据字典。
    
    Args:
        config: 联邦学习配置对象，包含数据集路径、类型、分割比例等参数
               详见 ``federatedscope.core.configs``
    
    Returns:
        tuple: 包含以下元素的元组
            - data_dict (dict): 联邦学习数据集字典，格式为：
              {'client_id': {'train': dataset, 'test': dataset, 'val': dataset}}
            - config: 更新后的配置对象（包含实际客户端数量）
    
    Raises:
        ValueError: 当指定的数据集名称不受支持时抛出异常
    
    Note:
        - 支持的数据集：'femnist', 'celeba'
        - 自动根据配置调整客户端数量
        - 数据集索引从 1 开始（client_id: 1, 2, 3, ...）
    """
    # 获取数据集分割比例配置
    splits = config.data.splits

    # 获取数据集根路径和类型
    path = config.data.root
    name = config.data.type.lower()
    
    # 构建数据变换函数（训练、验证、测试）
    transforms_funcs, val_transforms_funcs, test_transforms_funcs = \
        get_transform(config, 'torchvision')

    # 根据数据集类型加载相应的数据集
    if name in ['femnist', 'celeba']:
        dataset = LEAF_CV(root=path,
                          name=name,
                          s_frac=config.data.subsample,  # 子采样比例
                          tr_frac=splits[0],              # 训练集比例
                          val_frac=splits[1],             # 验证集比例
                          seed=1234,                      # 随机种子
                          **transforms_funcs)             # 数据变换函数
    else:
        raise ValueError(f'No dataset named: {name}!')

    # 确定实际的客户端数量（取配置值和数据集大小的最小值）
    client_num = min(len(dataset), config.federate.client_num
                     ) if config.federate.client_num > 0 else len(dataset)
    # 更新配置中的客户端数量
    config.merge_from_list(['federate.client_num', client_num])

    # 将数据集列表转换为客户端字典格式
    data_dict = dict()
    for client_idx in range(1, client_num + 1):
        # 注意：客户端ID从1开始，但数据集索引从0开始
        data_dict[client_idx] = dataset[client_idx - 1]

    return data_dict, config
