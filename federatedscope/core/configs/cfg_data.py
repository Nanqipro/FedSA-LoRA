"""数据配置模块

该模块定义了 FederatedScope 框架中数据相关的配置选项，包括数据集、数据加载器、
特征工程等各种配置参数。

主要功能:
    - 数据集配置: 支持多种数据集类型和分割策略
    - 数据加载器配置: 支持批次大小、打乱、多进程等选项
    - 特征工程配置: 支持特征选择、分箱、安全计算等
    - 异构任务配置: 支持多数据集联邦学习场景
    - 图数据配置: 支持 GraphSAINT、邻居采样等图神经网络数据加载
    - 安全配置: 支持加密和差分隐私保护

支持的数据集类型:
    - toy: 玩具数据集
    - files: 文件数据集
    - GLUE: 自然语言理解基准
    - 图数据集: 支持各种图神经网络数据
    - 异构数据集: 支持多任务联邦学习

支持的数据加载器:
    - base: 基础数据加载器
    - graphsaint-rw: GraphSAINT 随机游走数据加载器
    - neighbor: 邻居采样数据加载器
"""

# 导入日志模块
import logging

# 导入配置基类和注册函数
from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

# 获取日志记录器
logger = logging.getLogger(__name__)


def extend_data_cfg(cfg):
    """
    扩展数据相关的配置选项
    
    Args:
        cfg: 配置对象，用于添加数据相关的配置项
    """
    # ---------------------------------------------------------------------- #
    # 数据集相关选项
    # ---------------------------------------------------------------------- #
    cfg.data = CN()

    # 数据根目录
    cfg.data.root = 'data'
    # 数据集类型
    cfg.data.type = 'toy'
    # 用于 GLUE 中的 MNLI，是否匹配模式 [True, False]
    cfg.data.matched = True
    # 用于 GLUE，标签数量 [1, 2, 3]：1=回归，2=二分类，3=多分类
    cfg.data.num_labels = 2
    # 用于 GLUE，标签列表，如 ['0', '1', '2', '3', '4', '5']
    cfg.data.label_list = []
    # 是否保存生成的玩具数据
    cfg.data.save_data = False
    # 外部数据集的参数，如 [{'download': True}]
    cfg.data.args = []
    # 数据分割器类型
    cfg.data.splitter = ''
    # 分割器参数，如 [{'alpha': 0.5}]
    cfg.data.splitter_args = []
    # 服务器（索引为0的工作者）是否持有所有数据，在全局训练/评估情况下有用
    cfg.data.server_holds_all = False
    # 数据子采样比例
    cfg.data.subsample = 1.0
    # 训练、验证、测试数据分割比例
    cfg.data.splits = [0.8, 0.1, 0.1]
    # 如果为True，在分割过程中将保持客户端间训练/验证/测试集的标签分布一致
    cfg.data.consistent_label_distribution = True
    # cSBM（contextual Stochastic Block Model）参数
    cfg.data.cSBM_phi = [0.5, 0.5, 0.5]

    # 输入数据变换，如 [['ToTensor'], ['Normalize', {'mean': [0.9637], 'std': [0.1592]}]]
    cfg.data.transform = []
    # 目标数据变换，用法同上
    cfg.data.target_transform = []
    # torch_geometric 数据集的预变换，用法同上
    cfg.data.pre_transform = []

    # 如果未提供，则对所有分割使用 cfg.data.transform
    cfg.data.val_transform = []
    cfg.data.val_target_transform = []
    cfg.data.val_pre_transform = []
    cfg.data.test_transform = []
    cfg.data.test_target_transform = []
    cfg.data.test_pre_transform = []

    # 当 data.type = 'files' 时，data.file_path 生效
    cfg.data.file_path = ''

    # 数据加载器相关参数
    cfg.dataloader = CN()
    # 数据加载器类型
    cfg.dataloader.type = 'base'
    # 批次大小
    cfg.dataloader.batch_size = 64
    # 是否打乱数据
    cfg.dataloader.shuffle = True
    # 工作进程数
    cfg.dataloader.num_workers = 0
    # 是否丢弃最后一个不完整的批次
    cfg.dataloader.drop_last = False
    # 是否将数据加载到固定内存
    cfg.dataloader.pin_memory = False
    # GFL: GraphSAINT 数据加载器的游走长度
    cfg.dataloader.walk_length = 2
    # GFL: GraphSAINT 数据加载器的步数
    cfg.dataloader.num_steps = 30
    # GFL: 邻居采样器数据加载器的采样大小
    cfg.dataloader.sizes = [10, 5]
    # DP: -1 表示按评分隐私，否则按用户隐私
    cfg.dataloader.theta = -1

    # 二次函数相关配置
    cfg.data.quadratic = CN()
    # 二次函数维度
    cfg.data.quadratic.dim = 1
    # 最小曲率
    cfg.data.quadratic.min_curv = 0.02
    # 最大曲率
    cfg.data.quadratic.max_curv = 12.5

    # 异构NLP任务数据配置（用于ATC）
    # 多个数据集名称
    cfg.data.hetero_data_name = []
    # 每个数据集可以分割给多个客户端
    cfg.data.num_of_client_for_data = []
    # 最大序列长度
    cfg.data.max_seq_len = 384
    # 最大目标长度
    cfg.data.max_tgt_len = 128
    # 最大查询长度
    cfg.data.max_query_len = 128
    # 截断步长
    cfg.data.trunc_stride = 128
    # 缓存目录
    cfg.data.cache_dir = ''
    # 异构合成批次大小
    cfg.data.hetero_synth_batch_size = 32
    # 异构合成主要权重
    cfg.data.hetero_synth_prim_weight = 0.5
    # 异构合成特征维度
    cfg.data.hetero_synth_feat_dim = 128
    # 对比数量
    cfg.data.num_contrast = 0
    # 是否为调试模式
    cfg.data.is_debug = False

    # 特征工程配置
    cfg.feat_engr = CN()
    # 特征工程类型
    cfg.feat_engr.type = ''
    # 场景类型
    cfg.feat_engr.scenario = 'hfl'
    # 分箱数量（用于分箱）
    cfg.feat_engr.num_bins = 5
    # 选择阈值（用于特征选择）
    cfg.feat_engr.selec_threshold = 0.05
    # 选择WOE分箱方法
    cfg.feat_engr.selec_woe_binning = 'quantile'

    # 安全特征工程配置
    cfg.feat_engr.secure = CN()
    # 安全类型
    cfg.feat_engr.secure.type = 'encrypt'
    # 密钥大小
    cfg.feat_engr.secure.key_size = 3072

    # 加密配置
    cfg.feat_engr.secure.encrypt = CN()
    # 加密类型
    cfg.feat_engr.secure.encrypt.type = 'dummy'

    # 差分隐私配置（开发中）
    cfg.feat_engr.secure.dp = CN()

    # --------------- 过时的配置 ---------------
    # TODO: 删除此代码块
    cfg.data.loader = ''
    cfg.data.batch_size = 64
    cfg.data.shuffle = True
    cfg.data.num_workers = 0
    cfg.data.drop_last = False
    cfg.data.walk_length = 2
    cfg.data.num_steps = 30
    cfg.data.sizes = [10, 5]

    # --------------- 注册对应的检查函数 ----------
    cfg.register_cfg_check_fun(assert_data_cfg)


def assert_data_cfg(cfg):
    """
    数据配置的断言检查函数
    
    Args:
        cfg: 配置对象
        
    Raises:
        AssertionError: 当配置值不符合要求时抛出异常
    """
    # 检查 GraphSAINT 随机游走数据加载器配置
    if cfg.dataloader.type == 'graphsaint-rw':
        assert cfg.model.layer == cfg.dataloader.walk_length, '采样大小不匹配'
    
    # 检查邻居采样器数据加载器配置
    if cfg.dataloader.type == 'neighbor':
        assert cfg.model.layer == len(
            cfg.dataloader.sizes), '采样大小不匹配'
    
    # 检查外部数据配置
    if '@' in cfg.data.type:
        assert cfg.federate.client_num > 0, '使用外部数据时，`federate.client_num` 应大于 0'
        assert cfg.data.splitter, '使用外部数据时，`data.splitter` 不应为空'

    # 异构NLP任务数据检查
    if len(cfg.data.num_of_client_for_data) > 0:
        assert cfg.federate.client_num == \
               sum(cfg.data.num_of_client_for_data), '`federate.client_num` 应等于 `data.num_of_client_for_data` 的总和'

    # --------------------------------------------------------------------
    # 为了与旧版本 FS 兼容
    # TODO: 删除此代码块
    if cfg.data.loader != '':
        logger.warning('配置 `cfg.data.loader` 将在未来版本中移除，请使用 `cfg.dataloader.type` 替代。')
        cfg.dataloader.type = cfg.data.loader
    if cfg.data.batch_size != 64:
        logger.warning('配置 `cfg.data.batch_size` 将在未来版本中移除，请使用 `cfg.dataloader.batch_size` 替代。')
        cfg.dataloader.batch_size = cfg.data.batch_size
    if not cfg.data.shuffle:
        logger.warning('配置 `cfg.data.shuffle` 将在未来版本中移除，请使用 `cfg.dataloader.shuffle` 替代。')
        cfg.dataloader.shuffle = cfg.data.shuffle
    if cfg.data.num_workers != 0:
        logger.warning('配置 `cfg.data.num_workers` 将在未来版本中移除，请使用 `cfg.dataloader.num_workers` 替代。')
        cfg.dataloader.num_workers = cfg.data.num_workers
    if cfg.data.drop_last:
        logger.warning('配置 `cfg.data.drop_last` 将在未来版本中移除，请使用 `cfg.dataloader.drop_last` 替代。')
        cfg.dataloader.drop_last = cfg.data.drop_last
    if cfg.data.walk_length != 2:
        logger.warning('配置 `cfg.data.walk_length` 将在未来版本中移除，请使用 `cfg.dataloader.walk_length` 替代。')
        cfg.dataloader.walk_length = cfg.data.walk_length
    if cfg.data.num_steps != 30:
        logger.warning('配置 `cfg.data.num_steps` 将在未来版本中移除，请使用 `cfg.dataloader.num_steps` 替代。')
        cfg.dataloader.num_steps = cfg.data.num_steps
    if cfg.data.sizes != [10, 5]:
        logger.warning('配置 `cfg.data.sizes` 将在未来版本中移除，请使用 `cfg.dataloader.sizes` 替代。')
        cfg.dataloader.sizes = cfg.data.sizes
    # --------------------------------------------------------------------


# 注册数据配置扩展函数
register_config("data", extend_data_cfg)
