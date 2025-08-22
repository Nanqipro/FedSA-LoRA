"""FederatedScope 项目安装配置文件

该文件定义了 FederatedScope 联邦学习框架的安装配置，包括：
- 项目基本信息和版本
- 依赖包管理
- 不同功能模块的可选依赖
- 项目元数据和分类信息
"""

from __future__ import absolute_import, division, print_function

import setuptools

# 项目基本信息
__name__ = 'federatedscope'
__version__ = '0.3.0'
URL = 'https://github.com/alibaba/FederatedScope'

# 最小依赖包列表 - 核心功能所需的基础依赖
minimal_requires = [
    'numpy<1.23.0',          # 数值计算库
    'scikit-learn==1.0.2',   # 机器学习库
    'scipy==1.7.3',          # 科学计算库
    'pandas',                # 数据处理库
    'grpcio>=1.45.0',        # gRPC 通信框架
    'grpcio-tools',          # gRPC 工具
    'pyyaml>=5.1',           # YAML 配置文件解析
    'fvcore',                # Facebook 核心视觉库
    'iopath',                # 统一的 I/O 路径处理
    'wandb',                 # 实验跟踪和可视化
    'tensorboard',           # TensorBoard 可视化
    'tensorboardX',          # TensorBoard 扩展
    'pympler',               # 内存分析工具
    'protobuf==3.19.4',      # 协议缓冲区
    'matplotlib',            # 绘图库
    'dill',                  # 序列化库
]

# 测试相关依赖
test_requires = [
    'pytest',        # 测试框架
    'pytest-cov',    # 测试覆盖率
]

# 开发环境依赖
dev_requires = test_requires + ['pre-commit', 'networkx', 'matplotlib']

# 组织管理相关依赖
org_requires = [
    'paramiko==2.11.0',  # SSH 连接库
    'celery[redis]',     # 分布式任务队列
    'cmd2',              # 命令行工具
]

# 应用相关依赖 - 支持各种机器学习任务
app_requires = [
    'torch-geometric==2.0.4',  # 图神经网络库
    'nltk',                     # 自然语言处理工具包
    'transformers==4.16.2',     # Transformer 模型库
    'tokenizers==0.10.3',       # 分词器
    'datasets',                 # 数据集库
    'sentencepiece',            # 句子分词
    'textgrid',                 # 文本网格处理
    'typeguard',                # 类型检查
    'openml==0.12.2',           # OpenML 数据集接口
]

# 大语言模型相关依赖
llm_requires = [
    'tokenizers==0.19.1',      # 分词器（LLM 版本）
    'transformers==4.42.3',    # Transformer 模型库（LLM 版本）
    'accelerate==0.33.0',      # 模型加速库
    'peft==0.12.0',            # 参数高效微调库
    'sentencepiece==0.1.99',   # 句子分词（LLM 版本）
    'datasets==2.20.0',        # 数据集库（LLM 版本）
]

# 超参数优化基准测试依赖
benchmark_hpo_requires = [
    'configspace==0.5.0',   # 配置空间定义
    'hpbandster==0.7.4',    # 超参数优化库
    'smac==1.3.3',          # 序列模型算法配置
    'optuna==2.10.0',       # 超参数优化框架
]

# 异构迁移学习基准测试依赖
benchmark_htl_requires = [
    'learn2learn',  # 元学习库
]

# 完整功能依赖包
full_requires = org_requires + benchmark_hpo_requires + \
                benchmark_htl_requires + app_requires

# 读取项目说明文档
with open("README.md", "r", encoding='UTF-8') as fh:
    long_description = fh.read()

# 项目安装配置
setuptools.setup(
    name=__name__,
    version=__version__,
    author="Alibaba Damo Academy",
    author_email="jones.wz@alibaba-inc.com",
    description="Federated learning package",  # 联邦学习包
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    keywords=['deep-learning', 'federated-learning', 'benchmark'],  # 关键词：深度学习、联邦学习、基准测试
    packages=[
        package for package in setuptools.find_packages()
        if package.startswith(__name__)
    ],
    install_requires=minimal_requires,  # 基础依赖
    extras_require={  # 可选依赖组合
        'test': test_requires,      # 测试依赖
        'app': app_requires,        # 应用依赖
        'llm': llm_requires,        # 大语言模型依赖
        'org': org_requires,        # 组织管理依赖
        'dev': dev_requires,        # 开发依赖
        'hpo': benchmark_hpo_requires,  # 超参数优化依赖
        'htl': benchmark_htl_requires,  # 异构迁移学习依赖
        'full': full_requires       # 完整功能依赖
    },
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',  # 要求 Python 3.9 及以上版本
)
