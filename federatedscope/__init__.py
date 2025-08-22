"""FederatedScope 联邦学习框架

FederatedScope 是一个全面的联邦学习平台，支持：
- 多种联邦学习算法
- 跨模态和跨领域的联邦学习
- 大语言模型的联邦微调
- 图神经网络的联邦训练
- 隐私保护和安全聚合
- 自动超参数优化
- 异构系统支持

版本: 0.3.0
作者: Alibaba Damo Academy
许可证: Apache License 2.0
"""

from __future__ import absolute_import, division, print_function

# 框架版本号
__version__ = '0.3.0'


def _setup_logger():
    """设置 FederatedScope 框架的默认日志配置
    
    配置包括：
    - 日志格式：时间戳 + 模块名 + 行号 + 日志级别 + 消息
    - 输出到标准错误流
    - 禁用日志传播以避免重复输出
    """
    import logging

    # 定义日志格式
    logging_fmt = "%(asctime)s (%(module)s:%(lineno)d)" \
                  "%(levelname)s: %(message)s"
    
    # 创建 federatedscope 专用的日志器
    logger = logging.getLogger("federatedscope")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(logging_fmt))
    logger.addHandler(handler)
    logger.propagate = False  # 禁用向上级日志器传播


# 初始化日志配置
_setup_logger()
