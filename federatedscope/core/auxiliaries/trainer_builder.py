"""训练器构建器模块

该模块负责根据配置创建相应的训练器实例。
支持多种类型的训练器，用于不同的机器学习任务和联邦学习场景。

支持的训练器类型：
- 通用训练器：GeneralTorchTrainer, GeneralTFTrainer
- 计算机视觉：CVTrainer
- 自然语言处理：NLPTrainer, LLMTrainer, GLUETrainer
- 图神经网络：GraphMiniBatchTrainer, NodeFullBatchTrainer等
- 矩阵分解：MFTrainer
- 对比学习：CLTrainer
- 垂直联邦学习：VerticalTrainer
- 攻击相关：AttackerTrainer

主要功能：
1. 根据配置类型创建相应的训练器实例
2. 支持插件式的训练器包装（差分隐私、个性化、攻击等）
3. 处理不同后端（PyTorch、TensorFlow）的训练器
4. 支持注册机制的自定义训练器
5. 提供训练器的动态导入和实例化
"""

import logging
import importlib

import federatedscope.register as register
from federatedscope.core.trainers import Trainer

logger = logging.getLogger(__name__)

# 尝试导入贡献的训练器模块
try:
    from federatedscope.contrib.trainer import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.trainer`, some modules are not '
        f'available.')

# 训练器类型到类名的映射字典
TRAINER_CLASS_DICT = {
    "cvtrainer": "CVTrainer",                          # 计算机视觉训练器
    "nlptrainer": "NLPTrainer",                        # 自然语言处理训练器
    "graphminibatch_trainer": "GraphMiniBatchTrainer",  # 图神经网络小批量训练器
    "linkfullbatch_trainer": "LinkFullBatchTrainer",    # 链接预测全批量训练器
    "linkminibatch_trainer": "LinkMiniBatchTrainer",    # 链接预测小批量训练器
    "nodefullbatch_trainer": "NodeFullBatchTrainer",    # 节点分类全批量训练器
    "nodeminibatch_trainer": "NodeMiniBatchTrainer",    # 节点分类小批量训练器
    "flitplustrainer": "FLITPlusTrainer",              # FLIT+训练器
    "flittrainer": "FLITTrainer",                      # FLIT训练器
    "fedvattrainer": "FedVATTrainer",                  # FedVAT训练器
    "fedfocaltrainer": "FedFocalTrainer",              # FedFocal训练器
    "mftrainer": "MFTrainer",                          # 矩阵分解训练器
    "cltrainer": "CLTrainer",                          # 对比学习训练器
    "lptrainer": "LPTrainer",                          # 链接预测训练器
    "atc_trainer": "ATCTrainer",                       # ATC训练器
    "llmtrainer": "LLMTrainer",                        # 大语言模型训练器
    "gluetrainer": "GLUETrainer"                       # GLUE任务训练器
}


def get_trainer(model=None,
                data=None,
                device=None,
                config=None,
                only_for_eval=False,
                is_attacker=False,
                monitor=None):
    """创建训练器实例
    
    该函数根据配置创建相应的训练器实例，支持多种类型的训练器
    和插件式的功能扩展。

    参数:
        model: 联邦学习中使用的模型
        data: 联邦学习中使用的数据
        device (str): 训练设备（'cpu' 或 'gpu'）
        config: 联邦学习配置，参见 ``federatedscope.core.configs``
        only_for_eval (bool): 是否仅用于评估，如果为True则移除训练例程
        is_attacker (bool): 是否为攻击者客户端
        monitor: 监控器实例，用于观察评估和系统指标

    返回:
        训练器实例

    注意:
        训练器类型与对应类的映射关系：
        ==================================  ===========================
        训练器类型                          来源
        ==================================  ===========================
        ``general``                         \
        ``core.trainers.GeneralTorchTrainer`` 和 \
        ``core.trainers.GeneralTFTrainer``
        ``cvtrainer``                       ``cv.trainer.trainer.CVTrainer``
        ``nlptrainer``                      ``nlp.trainer.trainer.NLPTrainer``
        ``graphminibatch_trainer``          \
        ``gfl.trainer.graphtrainer.GraphMiniBatchTrainer``
        ``linkfullbatch_trainer``           \
        ``gfl.trainer.linktrainer.LinkFullBatchTrainer``
        ``linkminibatch_trainer``           \
        ``gfl.trainer.linktrainer.LinkMiniBatchTrainer``
        ``nodefullbatch_trainer``           \
        ``gfl.trainer.nodetrainer.NodeFullBatchTrainer``
        ``nodeminibatch_trainer``           \
        ``gfl.trainer.nodetrainer.NodeMiniBatchTrainer``
        ``flitplustrainer``                 \
        ``gfl.flitplus.trainer.FLITPlusTrainer``
        ``flittrainer``                     \
        ``gfl.flitplus.trainer.FLITTrainer``
        ``fedvattrainer``                   \
        ``gfl.flitplus.trainer.FedVATTrainer``
        ``fedfocaltrainer``                 \
        ``gfl.flitplus.trainer.FedFocalTrainer``
        ``mftrainer``                       \
        ``federatedscope.mf.trainer.MFTrainer``
        ``llmtrainer``                      \
        ``federatedscope.llm.trainer.LLMTrainer``
        ``gluetrainer``                     \
        ``federatedscope.glue.trainer.GLUETrainer``
        ==================================  ===========================
        
        包装器函数如下：
        ==================================  ===========================
        包装器函数                          来源
        ==================================  ===========================
        ``nbafl``                           \
        ``core.trainers.wrap_nbafl_trainer``
        ``sgdmf``                           ``mf.trainer.wrap_MFTrainer``
        ``pfedme``                          \
        ``core.trainers.wrap_pFedMeTrainer``
        ``ditto``                           ``core.trainers.wrap_DittoTrainer``
        ``fedem``                           ``core.trainers.FedEMTrainer``
        ``fedprox``                         \
        ``core.trainers.wrap_fedprox_trainer``
        ``attack``                          \
        ``attack.trainer.wrap_benignTrainer`` 和 \
        ``attack.auxiliary.attack_trainer_builder.wrap_attacker_trainer``
        ==================================  ===========================
    """
    # 处理通用训练器类型
    if config.trainer.type == 'general':
        if config.backend == 'torch':
            # 使用PyTorch后端的通用训练器
            from federatedscope.core.trainers import GeneralTorchTrainer
            trainer = GeneralTorchTrainer(model=model,
                                          data=data,
                                          device=device,
                                          config=config,
                                          only_for_eval=only_for_eval,
                                          monitor=monitor)
        elif config.backend == 'tensorflow':
            # 使用TensorFlow后端的通用训练器
            from federatedscope.core.trainers import GeneralTFTrainer
            trainer = GeneralTFTrainer(model=model,
                                       data=data,
                                       device=device,
                                       config=config,
                                       only_for_eval=only_for_eval,
                                       monitor=monitor)
        else:
            raise ValueError('Unsupported backend: {}'.format(config.backend))
    elif config.trainer.type == 'none':
        # 不使用训练器
        return None
    elif config.trainer.type.lower() in TRAINER_CLASS_DICT:
        # 根据训练器类型确定模块路径
        if config.trainer.type.lower() in ['cvtrainer']:
            dict_path = "federatedscope.cv.trainer.trainer"  # 计算机视觉训练器
        elif config.trainer.type.lower() in ['nlptrainer']:
            dict_path = "federatedscope.nlp.trainer.trainer"  # NLP训练器
        elif config.trainer.type.lower() in ['cltrainer', 'lptrainer']:
            dict_path = "federatedscope.cl.trainer.trainer"  # 对比学习训练器
        elif config.trainer.type.lower() in [
                'graphminibatch_trainer',
        ]:
            dict_path = "federatedscope.gfl.trainer.graphtrainer"  # 图神经网络训练器
        elif config.trainer.type.lower() in [
                'linkfullbatch_trainer', 'linkminibatch_trainer'
        ]:
            dict_path = "federatedscope.gfl.trainer.linktrainer"  # 链接预测训练器
        elif config.trainer.type.lower() in [
                'nodefullbatch_trainer', 'nodeminibatch_trainer'
        ]:
            dict_path = "federatedscope.gfl.trainer.nodetrainer"  # 节点分类训练器
        elif config.trainer.type.lower() in [
                'flitplustrainer', 'flittrainer', 'fedvattrainer',
                'fedfocaltrainer'
        ]:
            dict_path = "federatedscope.gfl.flitplus.trainer"  # FLIT系列训练器
        elif config.trainer.type.lower() in ['mftrainer']:
            dict_path = "federatedscope.mf.trainer.trainer"  # 矩阵分解训练器
        elif config.trainer.type.lower() in ['atc_trainer']:
            dict_path = "federatedscope.nlp.hetero_tasks.trainer"  # ATC训练器
        elif config.trainer.type.lower() in ['llmtrainer']:
            dict_path = "federatedscope.llm.trainer.trainer"  # 大语言模型训练器
        elif config.trainer.type.lower() in ['gluetrainer']:
            dict_path = "federatedscope.glue.trainer.trainer"  # GLUE任务训练器
        else:
            raise ValueError('Unknown trainer type: {}'.format(config.trainer.type))

        # 动态导入并实例化训练器类
        trainer_cls = getattr(importlib.import_module(name=dict_path),
                              TRAINER_CLASS_DICT[config.trainer.type.lower()])
        trainer = trainer_cls(model=model,
                              data=data,
                              device=device,
                              config=config,
                              only_for_eval=only_for_eval,
                              monitor=monitor)
    elif config.trainer.type.lower() in ['verticaltrainer']:
        # 处理垂直联邦学习训练器
        from federatedscope.vertical_fl.tree_based_models.trainer.utils \
            import get_vertical_trainer
        trainer = get_vertical_trainer(config=config,
                                       model=model,
                                       data=data,
                                       device=device,
                                       monitor=monitor)
    else:
        # 尝试查找用户注册的训练器
        trainer = None
        for func in register.trainer_dict.values():
            trainer_cls = func(config.trainer.type)
            if trainer_cls is not None:
                trainer = trainer_cls(model=model,
                                      data=data,
                                      device=device,
                                      config=config,
                                      only_for_eval=only_for_eval,
                                      monitor=monitor)
                break
        if trainer is None:
            raise ValueError('Trainer {} is not provided'.format(
                config.trainer.type))

    # 检查训练器是否继承自基础Trainer类
    if not isinstance(trainer, Trainer):
        logger.warning(f'Hook-like plug-in functions cannot be enabled when '
                       f'using {trainer}. If you want use our wrapper '
                       f'functions for your trainer please consider '
                       f'inheriting from '
                       f'`federatedscope.core.trainers.Trainer` instead.')
        return trainer

    # 差分隐私插件
    if config.nbafl.use:
        from federatedscope.core.trainers import wrap_nbafl_trainer
        trainer = wrap_nbafl_trainer(trainer)
    if config.sgdmf.use:
        from federatedscope.mf.trainer import wrap_MFTrainer
        trainer = wrap_MFTrainer(trainer)

    # 个性化联邦学习插件
    if config.federate.method.lower() == "pfedme":
        from federatedscope.core.trainers import wrap_pFedMeTrainer
        # 包装风格：实例a（类A）-> 实例a（类A）
        trainer = wrap_pFedMeTrainer(trainer)
    elif config.federate.method.lower() == "ditto":
        from federatedscope.core.trainers import wrap_DittoTrainer
        # 包装风格：实例a（类A）-> 实例a（类A）
        trainer = wrap_DittoTrainer(trainer)
    elif config.federate.method.lower() == "fedem":
        from federatedscope.core.trainers import FedEMTrainer
        # 复制构造风格：实例a（类A）-> 实例b（类B）
        trainer = FedEMTrainer(model_nums=config.model.model_num_per_trainer,
                               base_trainer=trainer)
    elif config.federate.method.lower() == "fedrep":
        from federatedscope.core.trainers import wrap_FedRepTrainer
        # 包装风格：实例a（类A）-> 实例a（类A）
        trainer = wrap_FedRepTrainer(trainer)

    # 攻击相关插件
    if 'backdoor' in config.attack.attack_method:
        from federatedscope.attack.trainer import wrap_benignTrainer
        trainer = wrap_benignTrainer(trainer)

    if is_attacker:
        if 'backdoor' in config.attack.attack_method:
            logger.info('--------This client is a backdoor attacker --------')
        else:
            logger.info('-------- This client is an privacy attacker --------')
        from federatedscope.attack.auxiliary.attack_trainer_builder \
            import wrap_attacker_trainer
        trainer = wrap_attacker_trainer(trainer, config)

    elif 'backdoor' in config.attack.attack_method:
        logger.info(
            '----- This client is a benign client for backdoor attacks -----')

    # 联邦算法插件
    if config.fedprox.use:
        from federatedscope.core.trainers import wrap_fedprox_trainer
        trainer = wrap_fedprox_trainer(trainer)

    # 微调相关插件
    if config.finetune.before_eval and config.finetune.simple_tuning:
        from federatedscope.core.trainers import wrap_Simple_tuning_Trainer
        trainer = wrap_Simple_tuning_Trainer(trainer)

    return trainer
