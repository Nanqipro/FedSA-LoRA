"""矩阵分解模型构建器

该模块提供了构建矩阵分解（Matrix Factorization）模型的功能。
主要功能包括：
- 构建垂直联邦学习矩阵分解模型（VMFNet）
- 构建水平联邦学习矩阵分解模型（HMFNet）
- 支持推荐系统和协同过滤任务

支持的模型类型：
- VMFNet：垂直矩阵分解网络，适用于垂直联邦学习场景
- HMFNet：水平矩阵分解网络，适用于水平联邦学习场景

应用场景：
- 推荐系统
- 协同过滤
- 评分预测
- 联邦学习环境下的矩阵分解
"""


def get_mfnet(model_config, data_shape):
    """根据模型配置构建矩阵分解模型

    根据配置参数构建相应的矩阵分解模型，支持垂直和水平联邦学习场景。

    参数:
        model_config: 模型配置对象，包含模型类型、用户数、物品数、隐藏层大小等参数
        data_shape (int): 输入数据的形状，在不同模型中表示不同含义：
                         - VMFNet中表示物品数量
                         - HMFNet中表示用户数量

    返回:
        构建好的矩阵分解模型实例

    注意:
        - VMFNet适用于垂直联邦学习，其中用户特征和物品特征分布在不同参与方
        - HMFNet适用于水平联邦学习，其中不同参与方拥有不同的用户子集
        - 模型类型通过model_config.type指定，不区分大小写
    """
    if model_config.type.lower() == 'vmfnet':
        # 构建垂直矩阵分解网络（Vertical Matrix Factorization Network）
        from federatedscope.mf.model.model import VMFNet
        return VMFNet(num_user=model_config.num_user,  # 用户数量
                      num_item=data_shape,  # 物品数量（从数据形状获取）
                      num_hidden=model_config.hidden)  # 隐藏层大小
    else:
        # 构建水平矩阵分解网络（Horizontal Matrix Factorization Network）
        from federatedscope.mf.model.model import HMFNet
        return HMFNet(num_user=data_shape,  # 用户数量（从数据形状获取）
                      num_item=model_config.num_item,  # 物品数量
                      num_hidden=model_config.hidden)  # 隐藏层大小
