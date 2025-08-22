"""联邦学习压缩配置模块

该模块定义了联邦学习中用于通信效率优化的压缩相关配置选项。
主要用于配置量化压缩方法，以减少客户端与服务器之间的通信开销。

主要功能：
- 配置量化压缩方法和参数
- 支持均匀量化等压缩算法
- 提供配置验证和错误检查
- 优化联邦学习通信效率

支持的压缩方法：
- none: 不使用压缩
- uniform: 均匀量化压缩

量化位数：
- 8位量化
- 16位量化
"""

import logging  # 日志记录

from federatedscope.core.configs.config import CN  # 配置节点
from federatedscope.register import register_config  # 配置注册器

logger = logging.getLogger(__name__)


def extend_compression_cfg(cfg):
    """
    扩展压缩相关的配置选项
    
    为联邦学习配置添加压缩相关的参数，主要用于优化通信效率。
    通过量化等方法减少模型参数传输的数据量。
    
    Args:
        cfg: 全局配置对象，将被扩展压缩相关配置
        
    配置项说明：
        quantization.method: 量化方法 ['none', 'uniform']
        quantization.nbits: 量化位数 [8, 16]
    """
    # ---------------------------------------------------------------------- #
    # 压缩配置选项（用于通信效率优化）
    # ---------------------------------------------------------------------- #
    cfg.quantization = CN()  # 量化压缩配置节点

    # 量化参数配置
    cfg.quantization.method = 'none'  # 量化方法：['none', 'uniform']
    cfg.quantization.nbits = 8  # 量化位数：[8, 16]

    # --------------- 注册对应的配置检查函数 ----------
    cfg.register_cfg_check_fun(assert_compression_cfg)


def assert_compression_cfg(cfg):
    """
    验证压缩配置的有效性
    
    检查量化方法和量化位数等压缩相关配置是否合法。
    对于无效配置会发出警告或抛出异常。
    
    Args:
        cfg: 需要验证的配置对象
        
    Raises:
        ValueError: 当量化位数配置无效时抛出异常
        
    注意事项：
        - 量化方法必须是 ['none', 'uniform'] 中的一种
        - 量化位数必须是 [8, 16] 中的一种
        - 无效的量化方法会被自动修正为 'none'
    """
    # 检查量化方法是否有效
    if cfg.quantization.method.lower() not in ['none', 'uniform']:
        logger.warning(
            f'量化方法应该是 ["none", "uniform"] 中的一种，'
            f'但得到了 "{cfg.quantization.method}"。将其修改为 "none"')

    # 检查量化位数是否有效（仅在启用量化时检查）
    if cfg.quantization.method.lower(
    ) != 'none' and cfg.quantization.nbits not in [8, 16]:
        raise ValueError(f'cfg.quantization.nbits 的值无效，'
                         f'应该是 [8, 16] 中的一种，但得到了 '
                         f'{cfg.quantization.nbits}。')


register_config("compression", extend_compression_cfg)
