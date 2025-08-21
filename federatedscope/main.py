# 导入必要的系统模块
import os
import sys

# 禁用 tokenizers 的并行处理，避免多进程冲突
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 开发模式标志，用于简化 federatedscope 的重新设置
DEV_MODE = False  # 当我们修改 federatedscope 源代码时，简化重新设置过程
if DEV_MODE:
    # 在开发模式下，将父目录添加到 Python 路径中
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

# 导入 FederatedScope 核心模块
from federatedscope.core.cmd_args import parse_args, parse_client_cfg  # 命令行参数解析
from federatedscope.core.auxiliaries.data_builder import get_data  # 数据构建器
from federatedscope.core.auxiliaries.utils import setup_seed, get_ds_rank  # 工具函数：随机种子设置和分布式排名
from federatedscope.core.auxiliaries.logging import update_logger  # 日志更新器
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls  # 客户端和服务器类构建器
from federatedscope.core.configs.config import global_cfg, CfgNode  # 全局配置和配置节点
from federatedscope.core.auxiliaries.runner_builder import get_runner  # 运行器构建器

# 清除代理设置，避免网络连接问题
if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

if __name__ == '__main__':
    # 克隆全局配置作为初始配置
    init_cfg = global_cfg.clone()
    # 解析命令行参数
    args = parse_args()
    # 如果指定了配置文件，则从文件中合并配置
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    # 解析客户端配置选项
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    # 将配置选项合并到初始配置中
    init_cfg.merge_from_list(cfg_opt)

    # 如果启用了 DeepSpeed，则初始化分布式训练
    if init_cfg.llm.deepspeed.use:
        import deepspeed
        deepspeed.init_distributed()

    # 更新日志配置，清除之前的日志处理器并设置分布式排名
    update_logger(init_cfg, clear_before_add=True, rank=get_ds_rank())
    # 设置随机种子以确保实验的可重复性
    setup_seed(init_cfg.seed)

    # 加载客户端配置文件
    if args.client_cfg_file:
        # 从文件中加载客户端配置
        client_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        # client_cfgs.set_new_allowed(True)  # 允许添加新的配置项（已注释）
        # 将客户端配置选项合并到配置中
        client_cfgs.merge_from_list(client_cfg_opt)
    else:
        client_cfgs = None

    # 联邦数据集可能会改变客户端数量
    # 因此，我们允许数据集创建过程修改全局配置对象
    data, modified_cfg = get_data(config=init_cfg.clone(),
                                  client_cfgs=client_cfgs)
    # 将数据构建过程中修改的配置合并到初始配置中
    init_cfg.merge_from_other_cfg(modified_cfg)

    # 如果指定了本地训练的客户端索引（非0），则进行单客户端训练
    if init_cfg.federate.client_idx_for_local_train != 0:
        # 设置客户端数量为1
        init_cfg.federate.client_num = 1
        # 重新组织数据，只保留服务器数据（索引0）和指定的客户端数据
        new_data = {0: data[0]} if 0 in data.keys() else dict()
        new_data[1] = data[init_cfg.federate.client_idx_for_local_train]
        data = new_data
    
    # 冻结配置，防止后续修改
    init_cfg.freeze()
    # 创建运行器，传入数据、服务器类、客户端类和配置
    runner = get_runner(data=data,
                        server_class=get_server_cls(init_cfg),  # 获取服务器类
                        client_class=get_client_cls(init_cfg),  # 获取客户端类
                        config=init_cfg.clone(),  # 克隆配置
                        client_configs=client_cfgs)  # 客户端配置
    # 运行联邦学习过程
    _ = runner.run()