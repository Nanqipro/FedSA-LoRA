"""模型配置模块

该模块定义了 FederatedScope 框架中模型相关的配置选项，包括各种深度学习模型、
损失函数和正则化器的配置参数。

主要功能:
    - 模型架构配置: 支持线性回归、神经网络、图神经网络等多种模型
    - 任务类型配置: 支持节点分类、图分类、推荐系统等任务
    - 模型参数配置: 隐藏层大小、dropout、层数等超参数
    - 树模型配置: 支持基于树的联邦学习算法
    - 语言模型配置: 支持 BERT 等预训练语言模型
    - 损失函数配置: 支持多种损失函数类型
    - 正则化配置: 支持各种正则化技术

支持的模型类型:
    - lr: 线性回归
    - mlp: 多层感知机
    - gcn: 图卷积网络
    - gat: 图注意力网络
    - sage: GraphSAGE
    - gin: 图同构网络
    - tree: 基于树的模型
    - bert: BERT 语言模型

支持的任务类型:
    - node: 节点分类
    - graph: 图分类
    - link: 链接预测
    - recommendation: 推荐系统
"""

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_model_cfg(cfg):
    """
    扩展模型相关的配置选项
    
    Args:
        cfg: 配置对象，用于添加模型相关的配置项
    """
    # ---------------------------------------------------------------------- #
    # 模型相关选项
    # ---------------------------------------------------------------------- #
    cfg.model = CN()

    # 每个训练器的模型数量（某些方法可能在每个训练器中使用多个模型）
    cfg.model.model_num_per_trainer = 1
    # 模型类型，如 'lr', 'mlp', 'gcn', 'gat' 等
    cfg.model.type = 'lr'
    # 是否使用偏置项
    cfg.model.use_bias = True
    # 任务类型，如 'node', 'graph', 'link' 等
    cfg.model.task = 'node'
    # 隐藏层大小
    cfg.model.hidden = 256
    # Dropout 概率
    cfg.model.dropout = 0.5
    # 输入通道数（如果为0，模型将根据数据形状构建）
    cfg.model.in_channels = 0
    # 输出通道数
    cfg.model.out_channels = 1
    # 网络层数（在 GPR-GNN 中，K = layer）
    cfg.model.layer = 2
    # 图池化方法，如 'mean', 'max', 'sum' 等
    cfg.model.graph_pooling = 'mean'
    # 嵌入向量大小
    cfg.model.embed_size = 8
    # 物品数量（用于推荐系统）
    cfg.model.num_item = 0
    # 用户数量（用于推荐系统）
    cfg.model.num_user = 0
    # 输入形状，元组格式，如 (in_channel, h, w)
    cfg.model.input_shape = ()

    # 基于树的模型配置
    # 正则化参数 lambda
    cfg.model.lambda_ = 0.1
    # 最小分割损失减少 gamma
    cfg.model.gamma = 0
    # 树的数量
    cfg.model.num_of_trees = 10
    # 最大树深度
    cfg.model.max_tree_depth = 3

    # 异构 NLP 任务的语言模型配置
    # 训练阶段，可选值: ['assign', 'contrast']
    cfg.model.stage = ''
    # 预训练模型类型
    cfg.model.model_type = 'google/bert_uncased_L-2_H-128_A-2'
    # 预训练任务列表
    cfg.model.pretrain_tasks = []
    # 下游任务列表
    cfg.model.downstream_tasks = []
    # 标签数量
    cfg.model.num_labels = 1
    # 最大序列长度
    cfg.model.max_length = 200
    # 最小序列长度
    cfg.model.min_length = 1
    # 不重复 n-gram 大小
    cfg.model.no_repeat_ngram_size = 3
    # 长度惩罚系数
    cfg.model.length_penalty = 2.0
    # beam search 的 beam 数量
    cfg.model.num_beams = 5
    # 标签平滑系数
    cfg.model.label_smoothing = 0.1
    # 最佳答案候选数量
    cfg.model.n_best_size = 20
    # 最大答案长度
    cfg.model.max_answer_len = 30
    # 空答案分数差异阈值
    cfg.model.null_score_diff_threshold = 0.0
    # 是否使用对比损失
    cfg.model.use_contrastive_loss = False
    # 对比学习的 top-k 值
    cfg.model.contrast_topk = 100
    # 对比学习的温度参数
    cfg.model.contrast_temp = 1.0

    # ---------------------------------------------------------------------- #
    # 损失函数相关选项
    # ---------------------------------------------------------------------- #
    cfg.criterion = CN()

    # 损失函数类型，如 'MSELoss', 'CrossEntropyLoss', 'BCELoss' 等
    cfg.criterion.type = 'MSELoss'

    # ---------------------------------------------------------------------- #
    # 正则化器相关选项
    # ---------------------------------------------------------------------- #
    cfg.regularizer = CN()

    # 正则化器类型，如 'L1', 'L2', 'elastic_net' 等
    cfg.regularizer.type = ''
    # 正则化强度参数
    cfg.regularizer.mu = 0.

    # --------------- 注册对应的检查函数 ----------
    cfg.register_cfg_check_fun(assert_model_cfg)


def assert_model_cfg(cfg):
    """
    模型配置的断言检查函数
    
    Args:
        cfg: 配置对象
        
    Note:
        目前暂无特定的模型配置检查逻辑
    """
    pass


register_config("model", extend_model_cfg)
