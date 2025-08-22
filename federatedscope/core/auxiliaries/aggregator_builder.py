"""聚合器构建器模块

该模块负责根据联邦学习方法和配置创建相应的聚合器实例。
聚合器是联邦学习中用于聚合所有客户端模型的协议组件。

支持的聚合方法：
- FedAvg：联邦平均算法
- FedOpt：联邦优化算法
- pFedMe：个性化联邦学习
- Ditto：个性化联邦学习
- FedSAGE+：图神经网络联邦学习
- GCFL+：图聚类联邦学习
- Krum：拜占庭容错聚合
- Median：中位数聚合
- Bulyan：拜占庭容错聚合
- TrimmedMean：修剪均值聚合
- NormBounding：范数约束聚合

主要功能：
1. 根据方法类型选择合适的聚合器
2. 支持在线和异步聚合模式
3. 处理不同后端（PyTorch/TensorFlow）
4. 支持鲁棒性聚合规则
5. 处理异构任务聚合
"""

import logging
from federatedscope.core.configs import constants

logger = logging.getLogger(__name__)


def get_aggregator(method, model=None, device=None, online=False, config=None):
    """构建聚合器实例
    
    该函数根据指定的联邦学习方法创建相应的聚合器，用于聚合所有客户端的模型。
    聚合器是联邦学习中的核心组件，负责将多个客户端的模型参数合并为全局模型。

    参数:
        method (str): 决定使用哪种聚合器的关键字
        model: 待聚合的模型对象
        device (str): 聚合模型的设备位置（'cpu'或'gpu'）
        online (bool): 是否使用在线聚合器
        config: 联邦学习配置对象，详见federatedscope.core.configs

    返回:
        aggregator: 聚合器实例（详见core.aggregator模块）

    注意:
        方法与聚合器的对应关系：
        ==================================  ===========================
        方法                                聚合器
        ==================================  ===========================
        ``tensorflow``                      ``cross_backends.FedAvgAggregator``
        ``local``                           \
        ``core.aggregators.NoCommunicationAggregator``
        ``global``                          \
        ``core.aggregators.NoCommunicationAggregator``
        ``fedavg``                          \
        ``core.aggregators.OnlineClientsAvgAggregator`` 或 \
        ``core.aggregators.AsynClientsAvgAggregator`` 或 \
        ``ClientsAvgAggregator``
        ``pfedme``                          \
        ``core.aggregators.ServerClientsInterpolateAggregator``
        ``ditto``                           \
        ``core.aggregators.OnlineClientsAvgAggregator`` 或 \
        ``core.aggregators.AsynClientsAvgAggregator`` 或 \
        ``ClientsAvgAggregator``
        ``fedsageplus``                     \
        ``core.aggregators.OnlineClientsAvgAggregator`` 或 \
        ``core.aggregators.AsynClientsAvgAggregator`` 或 \
        ``ClientsAvgAggregator``
        ``gcflplus``                        \
        ``core.aggregators.OnlineClientsAvgAggregator`` 或 \
        ``core.aggregators.AsynClientsAvgAggregator`` 或 \
        ``ClientsAvgAggregator``
        ``fedopt``                          \
        ``core.aggregators.FedOptAggregator``
        ==================================  ===========================
    """
    # 处理TensorFlow后端
    if config.backend == 'tensorflow':
        from federatedscope.cross_backends import FedAvgAggregator
        return FedAvgAggregator(model=model, device=device)
    else:
        # 导入PyTorch后端的各种聚合器
        from federatedscope.core.aggregators import ClientsAvgAggregator, \
            OnlineClientsAvgAggregator, ServerClientsInterpolateAggregator, \
            FedOptAggregator, NoCommunicationAggregator, \
            AsynClientsAvgAggregator, KrumAggregator, \
            MedianAggregator, TrimmedmeanAggregator, \
            BulyanAggregator,  NormboundingAggregator

    # 鲁棒性聚合规则映射表
    STR2AGG = {
        'fedavg': ClientsAvgAggregator,        # 联邦平均
        'krum': KrumAggregator,                # Krum拜占庭容错
        'median': MedianAggregator,            # 中位数聚合
        'bulyan': BulyanAggregator,            # Bulyan拜占庭容错
        'trimmedmean': TrimmedmeanAggregator,  # 修剪均值
        'normbounding': NormboundingAggregator # 范数约束
    }

    # 确定聚合器类型
    if method.lower() in constants.AGGREGATOR_TYPE:
        aggregator_type = constants.AGGREGATOR_TYPE[method.lower()]
    else:
        # 使用默认的客户端平均聚合器
        aggregator_type = "clients_avg"
        logger.warning(
            'Aggregator for method {} is not implemented. Will use default one'
            .format(method))

    # 处理异构NLP任务的特殊聚合器
    if config.data.type.lower() == 'hetero_nlp_tasks' and \
            not config.federate.atc_vanilla:
        from federatedscope.nlp.hetero_tasks.aggregator import ATCAggregator
        return ATCAggregator(model=model, config=config, device=device)

    # 根据聚合器类型创建相应实例
    if config.fedopt.use or aggregator_type == 'fedopt':
        # 联邦优化聚合器
        return FedOptAggregator(config=config, model=model, device=device)
    elif aggregator_type == 'clients_avg':
        if online:
            # 在线客户端平均聚合器
            return OnlineClientsAvgAggregator(
                model=model,
                device=device,
                config=config,
                src_device=device
                if config.federate.share_local_model else 'cpu')
        elif config.asyn.use:
            # 异步客户端平均聚合器
            return AsynClientsAvgAggregator(model=model,
                                            device=device,
                                            config=config)
        else:
            # 同步客户端平均聚合器（支持鲁棒性规则）
            if config.aggregator.robust_rule not in STR2AGG:
                logger.warning(
                    f'The specified {config.aggregator.robust_rule} aggregtion\
                    rule has not been supported, the vanilla fedavg algorithm \
                    will be used instead.')
            return STR2AGG.get(config.aggregator.robust_rule,
                               ClientsAvgAggregator)(model=model,
                                                     device=device,
                                                     config=config)

    elif aggregator_type == 'server_clients_interpolation':
        # 服务器-客户端插值聚合器（用于个性化联邦学习）
        return ServerClientsInterpolateAggregator(
            model=model,
            device=device,
            config=config,
            beta=config.personalization.beta)
    elif aggregator_type == 'no_communication':
        # 无通信聚合器（用于本地训练）
        return NoCommunicationAggregator(model=model,
                                         device=device,
                                         config=config)
    else:
        raise NotImplementedError(
            "Aggregator {} is not implemented.".format(aggregator_type))
