"""模型构建器模块

该模块负责根据配置文件构建和初始化各种机器学习模型，支持多种模型类型和后端：

支持的模型类型：
- 线性模型：逻辑回归（lr）
- 神经网络：多层感知机（mlp）、卷积神经网络（convnet2/5、vgg11）
- 循环神经网络：LSTM
- Transformer模型：各种预训练语言模型
- 图神经网络：GCN、SAGE、GPR、GAT、GIN、MPNN
- 矩阵分解：VMFNet、HMFNet
- 树模型：XGBoost、GBDT、随机森林
- 对比学习：SimCLR
- 大语言模型：各种LLM模型
- 异构任务模型：ATC模型
- 表格模型：二次模型

支持的后端：
- PyTorch
- TensorFlow

主要功能：
1. 从数据中提取输入形状
2. 根据配置构建相应的模型
3. 获取可训练参数名称
"""

import logging
import numpy as np
import federatedscope.register as register

logger = logging.getLogger(__name__)

# 尝试导入贡献的模型模块
try:
    from federatedscope.contrib.model import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.model`, some modules are not '
        f'available.')


def get_shape_from_data(data, model_config, backend='torch'):
    """从给定数据中提取输入形状，用于构建模型
    
    该函数能够处理多种数据格式和模型类型，自动推断输入形状：
    - 矩阵分解模型：返回行数或列数
    - 图神经网络：返回节点特征形状、标签数量和边特征数量
    - 异构任务模型：返回None
    - 其他模型：从数据中提取特征维度
    
    用户也可以通过 `data.input_shape` 直接指定形状。

    参数:
        data (ClientData): 用于本地训练或评估的数据
        model_config: 模型配置对象
        backend (str): 后端类型，'torch' 或 'tensorflow'

    返回:
        shape (tuple): 输入形状
    """
    # 处理特殊情况
    # 矩阵分解网络：VMFNet使用列数，HMFNet使用行数
    if model_config.type.lower() in ['vmfnet', 'hmfnet']:
        return data['train'].n_col if model_config.type.lower(
        ) == 'vmfnet' else data['train'].n_row
    # 图神经网络：需要节点特征、标签数量和边特征
    elif model_config.type.lower() in [
            'gcn', 'sage', 'gpr', 'gat', 'gin', 'mpnn'
    ] or model_config.type.startswith('gnn_'):
        num_label = data['num_label'] if 'num_label' in data else None
        num_edge_features = data['data'][
            'num_edge_features'] if model_config.type == 'mpnn' else None
        if model_config.task.startswith('graph'):
            # 图级任务
            data_representative = next(iter(data['train']))
            return data_representative.x.shape, num_label, num_edge_features
        else:
            # 节点/链接级任务
            return data['data'].x.shape, num_label, num_edge_features
    # 异构任务模型：不需要输入形状
    elif model_config.type.lower() in ['atc_model']:
        return None

    if isinstance(data, dict):
        keys = list(data.keys())
        if 'test' in keys:
            key_representative = 'test'
        elif 'val' in keys:
            key_representative = 'val'
        elif 'train' in keys:
            key_representative = 'train'
        elif 'data' in keys:
            key_representative = 'data'
        else:
            key_representative = keys[0]
            logger.warning(f'We chose the key {key_representative} as the '
                           f'representative key to extract data shape.')
        data_representative = data[key_representative]
    else:
        # 处理非字典格式的数据
        data_representative = data

    # 从数据中提取形状
    if isinstance(data_representative, dict):
        if 'x' in data_representative:
            shape = np.asarray(data_representative['x']).shape
            if len(shape) == 1:  # (batch, ) = (batch, 1)
                return 1
            else:
                return shape
    elif backend == 'torch':
        import torch
        # 处理PyTorch DataLoader
        if issubclass(type(data_representative), torch.utils.data.DataLoader):
            x, _ = next(iter(data_representative))
            if isinstance(x, list):
                return x[0].shape
            return x.shape
        else:
            try:
                x, _ = data_representative
                if isinstance(x, list):
                    return x[0].shape
                return x.shape
            except:
                raise TypeError('Unsupported data type.')
    elif backend == 'tensorflow':
        # TODO: 在此处理更多TensorFlow类型
        shape = data_representative['x'].shape
        if len(shape) == 1:  # (batch, ) = (batch, 1)
            return 1
        else:
            return shape


def get_model(config, local_data=None, backend='torch'):
    """构建待训练的模型实例
    
    该函数是模型构建的主入口，根据配置文件中的模型类型创建相应的模型实例。
    支持多种模型类型和后端框架。

    参数:
        config: 配置对象，包含模型相关配置
        local_data: 模型实例化所负责的给定数据（可选）
        backend (str): 后端选择，'torch' 或 'tensorflow'
        
    返回:
        model: 实例化的模型对象

    注意:
      内置模型类型和对应源码的键值对如下所示:
        ===================================  ==============================
        模型类型                              源码位置
        ===================================  ==============================
        ``lr``                               ``core.lr.LogisticRegression`` \
        或 ``cross_backends.LogisticRegression``
        ``mlp``                              ``core.mlp.MLP``
        ``quadratic``                        ``tabular.model.QuadraticModel``
        ``convnet2, convnet5, vgg11``        ``cv.model.get_cnn()``
        ``lstm``                             ``nlp.model.get_rnn()``
        ``{}@transformers``                  ``nlp.model.get_transformer()``
        ``gcn, sage, gpr, gat, gin, mpnn``   ``gfl.model.get_gnn()``
        ``vmfnet, hmfnet``                   \
        ``mf.model.model_builder.get_mfnet()``
        ===================================  ==============================
    """
    model_config = config.model

    # 确定输入形状
    if model_config.type.lower() in \
            ['xgb_tree', 'gbdt_tree', 'random_forest'] or \
            model_config.type.lower().endswith('_llm'):
        # 树模型和大语言模型不需要输入形状
        input_shape = None
    elif local_data is not None:
        # 从本地数据中提取输入形状
        input_shape = get_shape_from_data(local_data, model_config, backend)
    else:
        # 使用配置中指定的输入形状
        input_shape = model_config.input_shape

    if input_shape is None:
        logger.warning('The input shape is None. Please specify the '
                       '`data.input_shape`(a tuple) or give the '
                       'representative data to `get_model` if necessary')

    # 尝试使用注册的模型函数构建模型
    for func in register.model_dict.values():
        model = func(model_config, input_shape)
        if model is not None:
            return model

    # 根据模型类型构建相应的模型
    if model_config.type.lower() == 'lr':
        # 逻辑回归模型
        if backend == 'torch':
            from federatedscope.core.lr import LogisticRegression
            model = LogisticRegression(in_channels=input_shape[-1],
                                       class_num=model_config.out_channels)
        elif backend == 'tensorflow':
            from federatedscope.cross_backends import LogisticRegression
            model = LogisticRegression(in_channels=input_shape[-1],
                                       class_num=1,
                                       use_bias=model_config.use_bias)
        else:
            raise ValueError

    elif model_config.type.lower() == 'mlp':
        # 多层感知机模型
        from federatedscope.core.mlp import MLP
        model = MLP(channel_list=[input_shape[-1]] + [model_config.hidden] *
                    (model_config.layer - 1) + [model_config.out_channels],
                    dropout=model_config.dropout)

    elif model_config.type.lower() == 'quadratic':
        # 二次模型（用于表格数据）
        from federatedscope.tabular.model import QuadraticModel
        model = QuadraticModel(input_shape[-1], 1)

    elif model_config.type.lower() in ['convnet2', 'convnet5', 'vgg11']:
        # 卷积神经网络模型
        from federatedscope.cv.model import get_cnn
        model = get_cnn(model_config, input_shape)
    elif model_config.type.lower() in [
            'simclr', 'simclr_linear', "supervised_local", "supervised_fedavg"
    ]:
        # 对比学习模型（SimCLR）
        from federatedscope.cl.model import get_simclr
        model = get_simclr(model_config, input_shape)
        if model_config.type.lower().endswith('linear'):
            # 对于线性版本，冻结非线性层参数
            for name, value in model.named_parameters():
                if not name.startswith('linear'):
                    value.requires_grad = False
    elif model_config.type.lower() in ['lstm']:
        # 循环神经网络模型（LSTM）
        from federatedscope.nlp.model import get_rnn
        model = get_rnn(model_config, input_shape)
    elif model_config.type.lower().endswith('transformers'):
        # Transformer模型
        from federatedscope.nlp.model import get_transformer
        model = get_transformer(model_config, input_shape)
    # GLUE数据集的大语言模型
    elif config.data.type.lower().endswith('glue') and model_config.type.lower().endswith('_llm'):
        from federatedscope.glue.model import get_llm
        model = get_llm(config)
    elif model_config.type.lower().endswith('_llm'):
        # 大语言模型
        from federatedscope.llm.model import get_llm
        model = get_llm(config)
    elif model_config.type.lower() in [
            'gcn', 'sage', 'gpr', 'gat', 'gin', 'mpnn'
    ]:
        # 图神经网络模型
        from federatedscope.gfl.model import get_gnn
        model = get_gnn(model_config, input_shape)
    elif model_config.type.lower() in ['vmfnet', 'hmfnet']:
        # 矩阵分解网络模型
        from federatedscope.mf.model.model_builder import get_mfnet
        model = get_mfnet(model_config, input_shape)
    elif model_config.type.lower() in [
            'xgb_tree', 'gbdt_tree', 'random_forest'
    ]:
        # 树模型（XGBoost、GBDT、随机森林）
        from federatedscope.vertical_fl.tree_based_models.model.model_builder \
            import get_tree_model
        model = get_tree_model(model_config)
    elif model_config.type.lower() in ['atc_model']:
        # 异构任务模型（ATC）
        from federatedscope.nlp.hetero_tasks.model import ATCModel
        model = ATCModel(model_config)
    else:
        raise ValueError('Model {} is not provided'.format(model_config.type))
    return model


def get_trainable_para_names(model):
    """获取模型中可训练参数的名称集合
    
    参数:
        model: 模型对象
        
    返回:
        set: 可训练参数名称的集合
    """
    return set(dict(list(model.named_parameters())).keys())
