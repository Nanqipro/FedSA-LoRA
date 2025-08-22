"""指标构建器模块

该模块负责根据指标名称创建相应的指标计算器。
支持多种评估指标，用于衡量模型在不同任务上的性能。

支持的内置指标：
- 损失指标：loss, avg_loss, loss_regular
- 分类指标：acc, f1, roc_auc, ap
- 回归指标：rmse, mse
- 计数指标：total, correct
- 排序指标：hits@n
- 其他指标：imp_ratio, std

主要功能：
1. 根据指标名称列表创建指标计算器字典
2. 支持注册的自定义指标
3. 提供指标优化方向信息（越大越好或越小越好）
4. 处理异构NLP任务的特殊指标
5. 支持贡献模块的扩展指标
"""

import logging
import federatedscope.register as register
from federatedscope.nlp.hetero_tasks.metric import *  # 异构NLP任务指标

logger = logging.getLogger(__name__)

# 尝试导入贡献的指标模块
try:
    from federatedscope.contrib.metrics import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.metrics`, some modules are not '
        f'available.')


def get_metric(types):
    """创建指标计算器字典
    
    该函数根据指标名称列表创建指标计算器字典，其中键为指标名称，
    值为包含计算函数和优化方向的元组。

    参数:
        types (list): 指标名称列表

    返回:
        dict: 指标计算器字典，格式如下：
              {'loss': (eval_loss, False), 'acc': (eval_acc, True), ...}
              其中元组第一个元素是计算函数，第二个元素表示是否越大越好

    注意:
        内置指标与相关函数及优化方向的对应关系：
        =================  =============================================  =====
        指标名称            来源                                          \
        越大越好
        =================  =============================================  =====
        ``loss``           ``monitors.metric_calculator.eval_loss``       False
        ``avg_loss``       ``monitors.metric_calculator.eval_avg_loss``   False
        ``total``          ``monitors.metric_calculator.eval_total``      False
        ``correct``        ``monitors.metric_calculator.eval_correct``    True
        ``acc``            ``monitors.metric_calculator.eval_acc``        True
        ``ap``             ``monitors.metric_calculator.eval_ap``         True
        ``f1``             ``monitors.metric_calculator.eval_f1_score``   True
        ``roc_auc``        ``monitors.metric_calculator.eval_roc_auc``    True
        ``rmse``           ``monitors.metric_calculator.eval_rmse``       False
        ``mse``            ``monitors.metric_calculator.eval_mse``        False
        ``loss_regular``   ``monitors.metric_calculator.eval_regular``    False
        ``imp_ratio``      ``monitors.metric_calculator.eval_imp_ratio``  True
        ``std``            ``None``                                       False
        ``hits@{n}``       ``monitors.metric_calculator.eval_hits``       True
        =================  =============================================  =====
    """
    # 初始化指标字典
    metrics = dict()
    
    # 遍历注册的指标函数
    for func in register.metric_dict.values():
        res = func(types)
        if res is not None:
            # 解析指标名称、计算函数和优化方向
            name, metric, the_larger_the_better = res
            metrics[name] = metric, the_larger_the_better
    
    # 检查是否有未找到的指标
    for key in types:
        if key not in metrics.keys():
            logger.warning(f'eval.metrics `{key}` method not found!')
    
    return metrics
