# 导入配置基类和注册函数
from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_training_cfg(cfg):
    """
    扩展训练相关的配置选项
    
    Args:
        cfg: 配置对象，用于添加训练相关的配置项
    """
    # ---------------------------------------------------------------------- #
    # 训练器相关选项
    # ---------------------------------------------------------------------- #
    cfg.trainer = CN()

    # 训练器类型，默认为通用训练器
    cfg.trainer.type = 'general'

    # SAM (Sharpness-Aware Minimization) 优化器相关配置
    cfg.trainer.sam = CN()
    cfg.trainer.sam.adaptive = False  # 是否使用自适应SAM
    cfg.trainer.sam.rho = 1.0  # SAM的扰动半径参数
    cfg.trainer.sam.eta = .0  # SAM的学习率调整参数

    # 本地熵正则化相关配置
    cfg.trainer.local_entropy = CN()
    cfg.trainer.local_entropy.gamma = 0.03  # 熵正则化权重
    cfg.trainer.local_entropy.inc_factor = 1.0  # 增长因子
    cfg.trainer.local_entropy.eps = 1e-4  # 数值稳定性参数
    cfg.trainer.local_entropy.alpha = 0.75  # 平滑参数

    # ATC (Asynchronous Training Control) 相关配置 (TODO: 后续合并)
    cfg.trainer.disp_freq = 50  # 显示频率
    cfg.trainer.val_freq = 100000000  # 验证频率（跨批次）

    # ---------------------------------------------------------------------- #
    # 训练相关选项
    # ---------------------------------------------------------------------- #
    cfg.train = CN()

    # 本地更新步数
    cfg.train.local_update_steps = 1
    # 训练单位：'batch' 或 'epoch'
    cfg.train.batch_or_epoch = 'batch'
    # 数据并行设备ID列表（用于 torch.nn.DataParallel）
    cfg.train.data_para_dids = []

    # 优化器配置（允许添加新参数）
    cfg.train.optimizer = CN(new_allowed=True)
    cfg.train.optimizer.type = 'SGD'  # 优化器类型
    cfg.train.optimizer.lr = 0.1  # 学习率
    
    # VeRA (Vector-based Random Matrix Adaptation) 相关配置
    # 为分类头和适应层引入独立的学习率
    cfg.train.vera = CN(new_allowed=True)
    cfg.train.vera.lr_c = 0.01  # 分类头的学习率

    # 学习率调度器配置（可通过 cfg.train.scheduler.aa = 'bb' 添加新参数）
    cfg.train.scheduler = CN(new_allowed=True)
    cfg.train.scheduler.type = ''  # 调度器类型
    cfg.train.scheduler.warmup_ratio = 0.0  # 预热比例

    # 当模型过大时，用户可以使用半精度模型
    cfg.train.is_enable_half = False

    # ---------------------------------------------------------------------- #
    # 微调相关选项
    # ---------------------------------------------------------------------- #
    cfg.finetune = CN()

    # 是否在评估前进行微调
    cfg.finetune.before_eval = False
    # 微调的本地更新步数
    cfg.finetune.local_update_steps = 1
    # 微调单位：'batch' 或 'epoch'
    cfg.finetune.batch_or_epoch = 'epoch'
    # 冻结参数的名称模式
    cfg.finetune.freeze_param = ""

    # 微调优化器配置
    cfg.finetune.optimizer = CN(new_allowed=True)
    cfg.finetune.optimizer.type = 'SGD'  # 优化器类型
    cfg.finetune.optimizer.lr = 0.1  # 学习率

    # 微调学习率调度器配置
    cfg.finetune.scheduler = CN(new_allowed=True)
    cfg.finetune.scheduler.type = ''  # 调度器类型
    cfg.finetune.scheduler.warmup_ratio = 0.0  # 预热比例

    # 简单微调配置
    cfg.finetune.simple_tuning = False  # 是否使用简单微调，默认：False
    cfg.finetune.epoch_linear = 10  # 线性头训练轮数，默认：10
    cfg.finetune.lr_linear = 0.005  # 线性头训练学习率
    cfg.finetune.weight_decay = 0.0  # 权重衰减
    cfg.finetune.local_param = []  # 微调参数列表

    # ---------------------------------------------------------------------- #
    # 梯度相关选项
    # ---------------------------------------------------------------------- #
    cfg.grad = CN()
    # 梯度裁剪阈值，负数表示不进行梯度裁剪
    cfg.grad.grad_clip = -1.0
    # 梯度累积次数
    cfg.grad.grad_accum_count = 1

    # ---------------------------------------------------------------------- #
    # 早停相关选项
    # ---------------------------------------------------------------------- #
    cfg.early_stop = CN()

    # 耐心值（int）：监控指标上次改善后等待的轮数
    # 注意：实际检查轮数 = patience * cfg.eval.freq
    # 要禁用早停，请将 early_stop.patience 设置为 0
    cfg.early_stop.patience = 5
    # 增量（float）：监控指标的最小变化量，用于指示改善
    cfg.early_stop.delta = 0.0
    # 当连续 `patience` 轮没有改善时早停，模式可选：['mean', 'best']
    cfg.early_stop.improve_indicator_mode = 'best'

    # --------------- 注册对应的检查函数 ----------
    cfg.register_cfg_check_fun(assert_training_cfg)


def assert_training_cfg(cfg):
    """
    训练配置的断言检查函数
    
    Args:
        cfg: 配置对象
        
    Raises:
        ValueError: 当配置值不符合要求时抛出异常
    """
    # 检查训练单位配置
    if cfg.train.batch_or_epoch not in ['batch', 'epoch']:
        raise ValueError(
            "'cfg.train.batch_or_epoch' 的值必须从 ['batch', 'epoch'] 中选择。")

    # 检查微调单位配置
    if cfg.finetune.batch_or_epoch not in ['batch', 'epoch']:
        raise ValueError(
            "'cfg.finetune.batch_or_epoch' 的值必须从 ['batch', 'epoch'] 中选择。")

    # TODO: 这里应该不应该检查？
    # 检查后端配置
    if cfg.backend not in ['torch', 'tensorflow']:
        raise ValueError(
            "'cfg.backend' 的值必须从 ['torch', 'tensorflow'] 中选择。")
    # 检查 TensorFlow 后端的分布式模式
    if cfg.backend == 'tensorflow' and cfg.federate.mode == 'standalone':
        raise ValueError(
            "当后端为 tensorflow 时，我们只支持分布式模式运行")
    # 检查 TensorFlow 后端的 GPU 使用
    if cfg.backend == 'tensorflow' and cfg.use_gpu is True:
        raise ValueError(
            "当后端为 tensorflow 时，我们只支持 CPU 运行")

    # 检查微调步数配置
    if cfg.finetune.before_eval is False and cfg.finetune.local_update_steps\
            <= 0:
        raise ValueError(
            f"当采用微调时，请设置有效的本地微调步数，当前值为 {cfg.finetune.local_update_steps}")


# 注册训练配置扩展函数
register_config("fl_training", extend_training_cfg)
