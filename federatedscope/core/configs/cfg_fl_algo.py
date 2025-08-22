"""联邦学习算法配置模块

该模块定义了 FederatedScope 框架中各种联邦学习算法的配置选项，包括经典的
联邦学习算法、个性化联邦学习算法和图联邦学习算法等。

主要功能:
    - FedOpt 算法配置: 支持联邦优化算法
    - FedProx 算法配置: 支持近端联邦学习算法
    - FedSWA 算法配置: 支持随机权重平均算法
    - 个性化配置: 支持 pFedMe、Ditto、FedRep 等个性化算法
    - 图联邦学习配置: 支持 FedSage+、GCFL+、FLIT+ 等图联邦算法

支持的联邦学习算法:
    - FedAvg: 联邦平均算法（基础算法）
    - FedOpt: 联邦优化算法
    - FedProx: 近端联邦学习算法
    - FedSWA: 随机权重平均算法
    - pFedMe: 个性化联邦学习算法
    - Ditto: 公平个性化联邦学习算法
    - FedRep: 基于表示学习的个性化联邦学习
    - FedSage+: 图联邦学习算法
    - GCFL+: 图聚类联邦学习算法
    - FLIT+: 联邦学习与知识蒸馏算法

算法特点:
    - 支持同构和异构数据分布
    - 支持个性化和全局模型训练
    - 支持图数据的联邦学习
    - 支持多种优化策略和正则化技术
"""

from federatedscope.core.configs.config import CN
from federatedscope.core.configs.yacs_config import Argument
from federatedscope.register import register_config


def extend_fl_algo_cfg(cfg):
    """
    扩展联邦学习算法相关的配置选项
    
    Args:
        cfg: 配置对象，用于添加联邦学习算法相关的配置项
    """
    # ---------------------------------------------------------------------- #
    # FedOpt 相关选项，通用联邦学习算法
    # ---------------------------------------------------------------------- #
    cfg.fedopt = CN()

    # 是否使用 FedOpt 算法
    cfg.fedopt.use = False

    # FedOpt 优化器配置
    cfg.fedopt.optimizer = CN(new_allowed=True)
    # FedOpt 优化器类型
    cfg.fedopt.optimizer.type = Argument(
        'SGD', description="optimizer type for FedOPT")
    # FedOpt 优化器学习率
    cfg.fedopt.optimizer.lr = Argument(
        0.01, description="learning rate for FedOPT optimizer")
    # 是否使用学习率退火
    cfg.fedopt.annealing = False
    # 退火步长
    cfg.fedopt.annealing_step_size = 2000
    # 退火衰减因子
    cfg.fedopt.annealing_gamma = 0.5

    # ---------------------------------------------------------------------- #
    # FedProx 相关选项，近端联邦学习算法
    # ---------------------------------------------------------------------- #
    cfg.fedprox = CN()

    # 是否使用 FedProx 算法
    cfg.fedprox.use = False
    # 近端项系数，控制本地模型与全局模型的接近程度
    cfg.fedprox.mu = 0.

    # ---------------------------------------------------------------------- #
    # FedSWA 相关选项，随机权重平均 (Stochastic Weight Averaging)
    # ---------------------------------------------------------------------- #
    cfg.fedswa = CN()
    # 是否使用 FedSWA 算法
    cfg.fedswa.use = False
    # SWA 更新频率
    cfg.fedswa.freq = 10
    # 开始 SWA 的轮数
    cfg.fedswa.start_rnd = 30

    # ---------------------------------------------------------------------- #
    # 个性化相关选项，个性化联邦学习 (pFL)
    # ---------------------------------------------------------------------- #
    cfg.personalization = CN()

    # 客户端特有参数名称，例如 ['pre', 'post']
    cfg.personalization.local_param = []
    # 是否共享不可训练参数
    cfg.personalization.share_non_trainable_para = False
    # 本地更新步数（-1 表示使用默认值）
    cfg.personalization.local_update_steps = -1
    
    # 正则化权重:
    # 值越小，越强调个性化模型
    # 对于 Ditto，默认值=0.1，搜索空间为 [0.05, 0.1, 0.2, 1, 2]
    # 对于 pFedMe，默认值=15
    cfg.personalization.regular_weight = 0.1

    # 个性化学习率:
    # 1) 对于 pFedMe，用于使用 K 步近似计算 theta 的个性化学习率
    # 2) 0.0 表示在用户未指定有效学习率时使用 optimizer.lr 的值
    cfg.personalization.lr = 0.0

    # pFedMe 的本地近似步数
    cfg.personalization.K = 5
    # pFedMe 的平均移动参数
    cfg.personalization.beta = 1.0

    # FedRep 算法参数:
    # 特征提取器学习率
    cfg.personalization.lr_feature = 0.1
    # 线性头学习率
    cfg.personalization.lr_linear = 0.1
    # 特征提取器训练轮数
    cfg.personalization.epoch_feature = 1
    # 线性头训练轮数
    cfg.personalization.epoch_linear = 2
    # 权重衰减
    cfg.personalization.weight_decay = 0.0

    # ---------------------------------------------------------------------- #
    # FedSage+ 相关选项，图联邦学习 (GFL)
    # ---------------------------------------------------------------------- #
    cfg.fedsageplus = CN()

    # 生成器生成的节点数量
    cfg.fedsageplus.num_pred = 5
    # 生成器隐藏层维度
    cfg.fedsageplus.gen_hidden = 128
    # 隐藏图的比例
    cfg.fedsageplus.hide_portion = 0.5
    # 生成器的联邦训练轮数
    cfg.fedsageplus.fedgen_epoch = 200
    # 生成器的本地预训练轮数
    cfg.fedsageplus.loc_epoch = 1
    # 缺失节点数量损失的系数
    cfg.fedsageplus.a = 1.0
    # 特征损失的系数
    cfg.fedsageplus.b = 1.0
    # 分类损失的系数
    cfg.fedsageplus.c = 1.0

    # ---------------------------------------------------------------------- #
    # GCFL+ 相关选项，图聚类联邦学习 (GFL)
    # ---------------------------------------------------------------------- #
    cfg.gcflplus = CN()

    # 平均范数的边界
    cfg.gcflplus.EPS_1 = 0.05
    # 最大范数的边界
    cfg.gcflplus.EPS_2 = 0.1
    # 梯度序列的长度
    cfg.gcflplus.seq_length = 5
    # 是否标准化 DTW 距离
    cfg.gcflplus.standardize = False

    # ---------------------------------------------------------------------- #
    # FLIT+ 相关选项，联邦学习与知识蒸馏 (GFL)
    # ---------------------------------------------------------------------- #
    cfg.flitplus = CN()

    # 焦点损失中的 gamma 参数 (公式4)
    cfg.flitplus.tmpFed = 0.5
    # phi 中的 lambda 参数 (公式10)
    cfg.flitplus.lambdavat = 0.5
    # omega 中的 beta 参数 (公式12)
    cfg.flitplus.factor_ema = 0.8
    # 平衡 lossLocalLabel 和 lossLocalVAT 的权重
    cfg.flitplus.weightReg = 1.0

    # --------------- 注册对应的检查函数 ----------
    cfg.register_cfg_check_fun(assert_fl_algo_cfg)


def assert_fl_algo_cfg(cfg):
    """
    联邦学习算法配置的断言检查函数
    
    Args:
        cfg: 配置对象
        
    Raises:
        AssertionError: 当配置值不符合要求时抛出异常
    """
    # 检查个性化本地更新步数配置
    if cfg.personalization.local_update_steps == -1:
        # 默认使用与正常模式相同的步数
        cfg.personalization.local_update_steps = \
            cfg.train.local_update_steps
        cfg.personalization.local_update_steps = \
            cfg.train.local_update_steps

    # 检查个性化学习率配置
    if cfg.personalization.lr <= 0.0:
        # 默认使用与正常模式相同的学习率
        cfg.personalization.lr = cfg.train.optimizer.lr

    # 检查 FedSWA 配置
    if cfg.fedswa.use:
        assert cfg.fedswa.start_rnd < cfg.federate.total_round_num, \
            f'`cfg.fedswa.start_rnd` {cfg.fedswa.start_rnd} 必须小于 ' \
            f'`cfg.federate.total_round_num` ' \
            f'{cfg.federate.total_round_num}。'


register_config("fl_algo", extend_fl_algo_cfg)
