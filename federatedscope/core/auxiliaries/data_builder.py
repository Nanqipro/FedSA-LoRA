"""数据构建器模块

该模块负责根据配置文件构建和初始化数据集，支持多种数据类型和领域：
- 计算机视觉（CV）：FEMNIST、CelebA、torchvision数据集等
- 自然语言处理（NLP）：Shakespeare、SubReddit、Twitter、torchtext、huggingface数据集等
- 图数据：节点级（Cora、CiteSeer、PubMed等）、链接级（Epinions、Ciao、FB15k等）、图级（MUTAG、HIV、PROTEINS等）
- 表格数据：toy、synthetic、quadratic、OpenML数据集等
- 推荐系统：MovieLens、Netflix等
- 垂直联邦学习数据集

主要功能：
1. 数据集加载和预处理
2. 数据转换器选择和应用
3. 联邦学习数据格式转换
4. 分布式模式数据转换
"""

import logging

from importlib import import_module
from federatedscope.core.data.utils import RegexInverseMap, load_dataset, \
    convert_data_mode
from federatedscope.core.auxiliaries.utils import setup_seed

import federatedscope.register as register

logger = logging.getLogger(__name__)

# 尝试导入贡献的数据模块
try:
    from federatedscope.contrib.data import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.data`, some modules are not '
        f'available.')

# TODO: 添加 PyGNodeDataTranslator 和 PyGLinkDataTranslator
# TODO: 将分割器移动到 PyGNodeDataTranslator 和 PyGLinkDataTranslator

# 数据转换器映射表：定义不同数据集类型对应的转换器
TRANS_DATA_MAP = {
    # 基础数据转换器：用于标准数据集和分子数据集
    'BaseDataTranslator': [
        '.*?@.*?', 'hiv', 'proteins', 'imdb-binary', 'bbbp', 'tox21', 'bace',
        'sider', 'clintox', 'esol', 'freesolv', 'lipo', 'cifar4cl', 'cifar4lp'
    ],
    # 虚拟数据转换器：用于联邦学习专用数据集
    'DummyDataTranslator': [
        'toy', 'quadratic', 'femnist', 'celeba', 'shakespeare', 'twitter',
        'subreddit', 'synthetic', 'ciao', 'epinions', '.*?vertical_fl_data.*?',
        '.*?movielens.*?', '.*?netflix.*?', '.*?cikmcup.*?',
        'graph_multi_domain.*?', 'cora', 'citeseer', 'pubmed', 'dblp_conf',
        'dblp_org', 'csbm.*?', 'fb15k-237', 'wn18', 'adult', 'abalone',
        'credit', 'blog'
    ],
    # 原始数据转换器：用于异构NLP任务
    'RawDataTranslator': ['hetero_nlp_tasks'],
}
# 创建反向映射，用于根据数据集名称查找对应的转换器
DATA_TRANS_MAP = RegexInverseMap(TRANS_DATA_MAP, None)


def get_data(config, client_cfgs=None):
    """实例化数据集并根据需要更新配置
    
    该函数是数据构建的主入口，负责：
    1. 设置数据生成的随机种子
    2. 尝试使用注册的数据函数加载数据
    3. 从源文件加载数据集
    4. 应用数据转换器将集中式数据集转换为联邦学习格式
    5. 在分布式模式下转换数据格式
    6. 恢复用户指定的随机种子

    参数:
        config: 配置节点对象，包含数据相关配置
        client_cfgs: 客户端特定配置的字典（可选）
        
    返回:
        tuple: (数据集对象, 更新后的配置)

    注意:
      支持的 ``data.type`` 如下所示:
        ==================================  ===========================
        Data type                           Domain
        ==================================  ===========================
        FEMNIST	                            CV
        Celeba	                            CV
        ``${DNAME}@torchvision``	        CV
        Shakespeare	                        NLP
        SubReddit	                        NLP
        Twitter (Sentiment140)	            NLP
        ``${DNAME}@torchtext``	            NLP
        ``${DNAME}@huggingface_datasets``  	NLP
        Cora	                            Graph (node-level)
        CiteSeer	                        Graph (node-level)
        PubMed	                            Graph (node-level)
        DBLP_conf	                        Graph (node-level)
        DBLP_org	                        Graph (node-level)
        csbm	                            Graph (node-level)
        Epinions	                        Graph (link-level)
        Ciao	                            Graph (link-level)
        FB15k	                            Graph (link-level)
        FB15k-237	                        Graph (link-level)
        WN18	                            Graph (link-level)
        MUTAG	                            Graph (graph-level)
        BZR	                                Graph (graph-level)
        COX2	                            Graph (graph-level)
        DHFR	                            Graph (graph-level)
        PTC_MR	                            Graph (graph-level)
        AIDS	                            Graph (graph-level)
        NCI1	                            Graph (graph-level)
        ENZYMES	                            Graph (graph-level)
        DD	                                Graph (graph-level)
        PROTEINS	                        Graph (graph-level)
        COLLAB	                            Graph (graph-level)
        IMDB-BINARY	                        Graph (graph-level)
        IMDB-MULTI	                        Graph (graph-level)
        REDDIT-BINARY	                    Graph (graph-level)
        HIV	                                Graph (graph-level)
        ESOL	                            Graph (graph-level)
        FREESOLV	                        Graph (graph-level)
        LIPO	                            Graph (graph-level)
        PCBA	                            Graph (graph-level)
        MUV	                                Graph (graph-level)
        BACE	                            Graph (graph-level)
        BBBP	                            Graph (graph-level)
        TOX21	                            Graph (graph-level)
        TOXCAST	                            Graph (graph-level)
        SIDER	                            Graph (graph-level)
        CLINTOX	                            Graph (graph-level)
        graph_multi_domain_mol	            Graph (graph-level)
        graph_multi_domain_small	        Graph (graph-level)
        graph_multi_domain_biochem	        Graph (graph-level)
        cikmcup	                            Graph (graph-level)
        toy	                                Tabular
        synthetic	                        Tabular
        quadratic	                        Tabular
        ``${DNAME}openml``	                Tabular
        vertical_fl_data	                Tabular(vertical)
        VFLMovieLens1M	                    Recommendation
        VFLMovieLens10M	                    Recommendation
        HFLMovieLens1M	                    Recommendation
        HFLMovieLens10M	                    Recommendation
        VFLNetflix	                        Recommendation
        HFLNetflix	                        Recommendation
        ==================================  ===========================
    """
    # 为数据生成固定随机种子，确保数据划分的可重复性
    setup_seed(12345)

    # 尝试使用注册的数据函数加载数据
    for func in register.data_dict.values():
        data_and_config = func(config, client_cfgs)
        if data_and_config is not None:
            return data_and_config

    # 从源文件加载数据集
    dataset, modified_config = load_dataset(config, client_cfgs)
    
    # 对非联邦学习数据集应用转换器，将其转换为联邦学习对应格式
    if dataset is not None:
        # 根据数据类型选择相应的转换器
        # 例如：LLM和GLUE数据集使用BaseDataTranslator
        translator = getattr(import_module('federatedscope.core.data'),
                             DATA_TRANS_MAP[config.data.type.lower()])(
                                 modified_config, client_cfgs)
        # 将集中式数据集转换为StandaloneDataDict格式
        data = translator(dataset)

        # 在分布式模式下将StandaloneDataDict转换为ClientData
        data = convert_data_mode(data, modified_config)
    else:
        data = None

    # 数据生成完成后恢复用户指定的随机种子
    setup_seed(config.seed)

    return data, modified_config
