# 导入必要的模块
import logging  # 日志记录

from federatedscope.core.configs.config import CN  # 配置节点类
from federatedscope.register import register_config  # 配置注册机制
import torch  # PyTorch 框架

# 获取日志记录器
logger = logging.getLogger(__name__)


def extend_fl_setting_cfg(cfg):
    """
    扩展联邦学习设置相关的配置选项
    
    Args:
        cfg: 全局配置对象
    """
    # ---------------------------------------------------------------------- #
    # 联邦学习相关配置选项
    # ---------------------------------------------------------------------- #
    cfg.federate = CN()

    # 客户端数量和索引配置
    cfg.federate.client_num = 0  # 总客户端数量
    cfg.federate.client_idx_for_local_train = 0  # 用于本地训练的客户端索引
    cfg.federate.sample_client_num = -1  # 每轮采样的客户端数量
    cfg.federate.sample_client_rate = -1.0  # 每轮采样的客户端比例
    cfg.federate.unseen_clients_rate = 0.0  # 未见过的客户端比例
    
    # 训练轮次和模式配置
    cfg.federate.total_round_num = 50  # 总训练轮数
    cfg.federate.mode = 'standalone'  # 运行模式：standalone（单机）或 distributed（分布式）
    
    # 模型和聚合配置
    cfg.federate.share_local_model = False  # 是否共享本地模型
    cfg.federate.data_weighted_aggr = False  # 是否使用数据加权聚合
    # 如果为 True，聚合权重为数据集中的训练样本数量
    cfg.federate.online_aggr = False  # 是否使用在线聚合
    cfg.federate.make_global_eval = False  # 是否进行全局评估
    cfg.federate.use_diff = False  # 是否使用差分更新
    
    # 数据合并配置（用于高效仿真）
    cfg.federate.merge_test_data = False  # 是否合并测试数据进行全局评估
    # 而不是在每个客户端执行测试
    cfg.federate.merge_val_data = False  # 是否合并验证数据
    # 仅在 merge_test_data 为 True 时启用，同样用于高效仿真

    # 联邦学习算法配置
    # 方法名用于内部确定不同聚合器、消息、处理器等的组合
    cfg.federate.method = "FedAvg"  # 联邦学习算法名称
    cfg.federate.ignore_weight = False  # 是否忽略聚合权重
    cfg.federate.freeze_A = False  # 是否冻结 LoRA A 矩阵（用于 FedSA-LoRA）
    cfg.federate.use_ss = False  # 是否应用秘密共享（Secret Sharing）
    
    # 模型保存和恢复配置
    cfg.federate.restore_from = ''  # 模型恢复路径
    cfg.federate.save_to = ''  # 模型保存路径
    cfg.federate.save_freq = -1  # 保存频率（-1 表示不保存）
    cfg.federate.save_client_model = False  # 是否保存客户端模型
    
    # 客户端参与配置
    cfg.federate.join_in_info = []  # 客户端加入时的信息要求（来自服务器）
    cfg.federate.sampler = 'uniform'  # 每轮训练中的客户端采样策略
    # 可选值：['uniform', 'group']
    cfg.federate.resource_info_file = ""  # 设备信息文件路径
    # 用于记录计算和通信能力

    # 单机模式下的并行配置
    cfg.federate.process_num = 1  # 进程数量
    cfg.federate.master_addr = '127.0.0.1'  # PyTorch 分布式训练的主节点地址
    cfg.federate.master_port = 29500  # PyTorch 分布式训练的主节点端口

    # ATC 相关配置（TODO: 后续合并）
    cfg.federate.atc_vanilla = False  # 是否使用原始 ATC
    cfg.federate.atc_load_from = ''  # ATC 加载路径

    # ---------------------------------------------------------------------- #
    # 分布式训练相关配置选项
    # ---------------------------------------------------------------------- #
    cfg.distribute = CN()

    # 基本分布式配置
    cfg.distribute.use = False  # 是否使用分布式训练
    cfg.distribute.server_host = '0.0.0.0'  # 服务器主机地址
    cfg.distribute.server_port = 50050  # 服务器端口
    cfg.distribute.client_host = '0.0.0.0'  # 客户端主机地址
    cfg.distribute.client_port = 50050  # 客户端端口
    cfg.distribute.role = 'client'  # 角色：'client' 或 'server'
    
    # 数据配置
    cfg.distribute.data_file = 'data'  # 数据文件名
    cfg.distribute.data_idx = -1  # 数据索引
    # data_idx 用于在分布式模式下指定数据索引，当采用集中式数据集进行仿真时
    # 格式为 {data_idx: data/dataloader}
    # data_idx = -1 表示参与者拥有整个数据集
    # 当 data_idx 为除 -1 外的其他无效值时，我们随机采样 data_idx 进行仿真
    
    # gRPC 通信配置
    cfg.distribute.grpc_max_send_message_length = 300 * 1024 * 1024  # 最大发送消息长度（300M）
    cfg.distribute.grpc_max_receive_message_length = 300 * 1024 * 1024  # 最大接收消息长度（300M）
    cfg.distribute.grpc_enable_http_proxy = False  # 是否启用 HTTP 代理
    cfg.distribute.grpc_compression = 'nocompression'  # 压缩方式：[deflate, gzip, nocompression]

    # ---------------------------------------------------------------------- #
    # 垂直联邦学习相关配置选项（用于演示）
    # ---------------------------------------------------------------------- #
    cfg.vertical = CN()
    
    # 基本垂直联邦学习配置
    cfg.vertical.use = False  # 是否使用垂直联邦学习
    cfg.vertical.mode = 'feature_gathering'  # 模式
    # 可选值：['feature_gathering', 'label_scattering']
    cfg.vertical.dims = [5, 10]  # 特征维度分配
    # 客户端 1 拥有前 5 个特征，客户端 2 拥有后 5 个特征
    
    # 加密和算法配置
    cfg.vertical.encryption = 'paillier'  # 加密方法
    cfg.vertical.key_size = 3072  # 密钥大小
    cfg.vertical.algo = 'lr'  # 算法类型
    # 可选值：['lr', 'xgb', 'gbdt', 'rf']
    cfg.vertical.feature_subsample_ratio = 1.0  # 特征子采样比例
    
    # 隐私保护配置
    cfg.vertical.protect_object = ''  # 保护对象
    # 可选值：[feature_order, grad_and_hess]
    cfg.vertical.protect_method = ''  # 保护方法
    # 对于 protect_object = feature_order：[dp, op_boost]
    # 对于 protect_object = grad_and_hess：[he]
    cfg.vertical.protect_args = []  # 保护参数
    # 'dp' 的默认值：{'bucket_num':100, 'epsilon':None}
    # 'op_boost' 的默认值：{'algo':'global', 'lower_bound':1,
    #                                 'upper_bound':100, 'epsilon':2}
    cfg.vertical.eval_protection = ''  # ['', 'he']
    cfg.vertical.data_size_for_debug = 0  # use a subset for debug in vfl,
    # 0 indicates using the entire dataset (disable debug mode)

    cfg.adapter = CN()
    cfg.adapter.use = False
    cfg.adapter.args = []

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_fl_setting_cfg)


def assert_fl_setting_cfg(cfg):
    assert cfg.federate.mode in ["standalone", "distributed"], \
        f"Please specify the cfg.federate.mode as the string standalone or " \
        f"distributed. But got {cfg.federate.mode}."

    # =============  client num related  ==============
    assert not (cfg.federate.client_num == 0
                and cfg.federate.mode == 'distributed'
                ), "Please configure the cfg.federate. in distributed mode. "

    assert 0 <= cfg.federate.unseen_clients_rate < 1, \
        "You specified in-valid cfg.federate.unseen_clients_rate"
    if 0 < cfg.federate.unseen_clients_rate < 1 and cfg.federate.method in [
            "local", "global"
    ]:
        logger.warning(
            "In local/global training mode, the unseen_clients_rate is "
            "in-valid, plz check your config")
        unseen_clients_rate = 0.0
        cfg.federate.unseen_clients_rate = unseen_clients_rate
    else:
        unseen_clients_rate = cfg.federate.unseen_clients_rate
    participated_client_num = max(
        1, int((1 - unseen_clients_rate) * cfg.federate.client_num))

    # sample client num pre-process
    sample_client_num_valid = (
        0 < cfg.federate.sample_client_num <=
        cfg.federate.client_num) and cfg.federate.client_num != 0
    sample_client_rate_valid = (0 < cfg.federate.sample_client_rate <= 1)

    sample_cfg_valid = sample_client_rate_valid or sample_client_num_valid
    non_sample_case = cfg.federate.method in ["local", "global"]
    if non_sample_case and sample_cfg_valid:
        logger.warning("In local/global training mode, "
                       "the sampling related configs are in-valid, "
                       "we will use all clients. ")

    if cfg.federate.method == "global":
        logger.info(
            "In global training mode, we will put all data in a proxy client. "
        )
        if cfg.federate.make_global_eval:
            cfg.federate.make_global_eval = False
            logger.warning(
                "In global training mode, we will conduct global evaluation "
                "in a proxy client rather than the server. The configuration "
                "cfg.federate.make_global_eval will be False.")

    if non_sample_case or not sample_cfg_valid:
        # (a) use all clients
        # in standalone mode, federate.client_num may be modified from 0 to
        # num_of_all_clients after loading the data
        if cfg.federate.client_num != 0:
            cfg.federate.sample_client_num = participated_client_num
    else:
        # (b) sampling case
        if sample_client_rate_valid:
            # (b.1) use sample_client_rate
            old_sample_client_num = cfg.federate.sample_client_num
            cfg.federate.sample_client_num = max(
                1,
                int(cfg.federate.sample_client_rate * participated_client_num))
            if sample_client_num_valid:
                logger.warning(
                    f"Users specify both valid sample_client_rate as"
                    f" {cfg.federate.sample_client_rate} "
                    f"and sample_client_num as {old_sample_client_num}.\n"
                    f"\t\tWe will use the sample_client_rate value to "
                    f"calculate "
                    f"the actual number of participated clients as"
                    f" {cfg.federate.sample_client_num}.")
            # (b.2) use sample_client_num, commented since the below two
            # lines do not change anything
            # elif sample_client_num_valid:
            #     cfg.federate.sample_client_num = \
            #     cfg.federate.sample_client_num

    if cfg.federate.use_ss:
        assert cfg.federate.client_num == cfg.federate.sample_client_num, \
            "Currently, we support secret sharing only in " \
            "all-client-participation case"

        assert cfg.federate.method != "local", \
            "Secret sharing is not supported in local training mode"

    # =============   aggregator related   ================
    assert (not cfg.federate.online_aggr) or (
        not cfg.federate.use_ss
    ), "Have not supported to use online aggregator and secrete sharing at " \
       "the same time"

    assert not cfg.federate.merge_test_data or (
            cfg.federate.merge_test_data and cfg.federate.mode == 'standalone'
    ), "The operation of merging test data can only used in standalone for " \
       "efficient simulation, please change 'federate.merge_test_data' to " \
       "False or change 'federate.mode' to 'distributed'."
    if cfg.federate.merge_test_data and not cfg.federate.make_global_eval:
        cfg.federate.make_global_eval = True
        logger.warning('Set cfg.federate.make_global_eval=True since '
                       'cfg.federate.merge_test_data=True')

    if cfg.federate.process_num > 1 and cfg.federate.mode != 'standalone':
        cfg.federate.process_num = 1
        logger.warning('Parallel training can only be used in standalone mode'
                       ', thus cfg.federate.process_num is modified to 1')
    if cfg.federate.process_num > 1 and not torch.cuda.is_available():
        cfg.federate.process_num = 1
        logger.warning(
            'No GPU found for your device, set cfg.federate.process_num=1')
    if torch.cuda.device_count() < cfg.federate.process_num:
        cfg.federate.process_num = torch.cuda.device_count()
        logger.warning(
            'We found the number of gpu is insufficient, '
            f'thus cfg.federate.process_num={cfg.federate.process_num}')
    # TODO
    if cfg.vertical.use:
        if cfg.vertical.algo == 'lr' and hasattr(cfg, "trainer") and \
                cfg.trainer.type != 'none':
            logger.warning(f"When given cfg.vertical.algo = 'lr', the value "
                           f"of cfg.trainer.type is expected to be 'none' "
                           f"but got {cfg.trainer.type}. Therefore "
                           f"cfg.trainer.type is changed to 'none' here")
            cfg.trainer.type = 'none'
        if cfg.vertical.algo == 'lr' and hasattr(cfg, "model") and \
                cfg.model.type != 'lr':
            logger.warning(f"When given cfg.vertical.algo = 'lr', the value "
                           f"of cfg.model.type is expected to be 'lr' "
                           f"but got {cfg.model.type}. Therefore "
                           f"cfg.model.type is changed to 'lr' here")
            cfg.model.type = 'lr'
        if cfg.vertical.algo in ['xgb', 'gbdt'] and hasattr(cfg, "trainer") \
                and cfg.trainer.type.lower() != 'verticaltrainer':
            logger.warning(
                f"When given cfg.vertical.algo = 'xgb' or 'gbdt', the value "
                f"of cfg.trainer.type is expected to be "
                f"'verticaltrainer' but got {cfg.trainer.type}. "
                f"Therefore cfg.trainer.type is changed to "
                f"'verticaltrainer' here")
            cfg.trainer.type = 'verticaltrainer'
        if cfg.vertical.algo == 'xgb' and hasattr(cfg, "model") and \
                cfg.model.type != 'xgb_tree':
            logger.warning(f"When given cfg.vertical.algo = 'xgb', the value "
                           f"of cfg.model.type is expected to be 'xgb_tree' "
                           f"but got {cfg.model.type}. Therefore "
                           f"cfg.model.type is changed to 'xgb_tree' here")
            cfg.model.type = 'xgb_tree'
        elif cfg.vertical.algo == 'gbdt' and hasattr(cfg, "model") and \
                cfg.model.type != 'gbdt_tree':
            logger.warning(f"When given cfg.vertical.algo = 'gbdt', the value "
                           f"of cfg.model.type is expected to be 'gbdt_tree' "
                           f"but got {cfg.model.type}. Therefore "
                           f"cfg.model.type is changed to 'gbdt_tree' here")
            cfg.model.type = 'gbdt_tree'

        if not (cfg.vertical.feature_subsample_ratio > 0
                and cfg.vertical.feature_subsample_ratio <= 1.0):
            raise ValueError(f'The value of vertical.feature_subsample_ratio '
                             f'must be in (0, 1.0], but got '
                             f'{cfg.vertical.feature_subsample_ratio}')

    if cfg.distribute.use and cfg.distribute.grpc_compression.lower() not in [
            'nocompression', 'deflate', 'gzip'
    ]:
        raise ValueError(f'The type of grpc compression is expected to be one '
                         f'of ["nocompression", "deflate", "gzip"], but got '
                         f'{cfg.distribute.grpc_compression}.')


register_config("fl_setting", extend_fl_setting_cfg)
