"""FederatedScope 超参数优化 (HPO) 主入口文件

该文件是 FederatedScope 框架中超参数优化功能的主入口点，提供：
- 超参数优化任务的配置和启动
- 自动调优调度器的初始化和运行
- 实验名称的自动生成
- 客户端配置的加载和合并

使用方法：
    python hpo.py --cfg path/to/config.yaml
"""

import os
import sys

# 开发模式标志 - 简化每次修改源代码后重新设置 federatedscope 的过程
DEV_MODE = False
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

# 导入核心模块
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.configs.config import global_cfg, CfgNode
from federatedscope.autotune import get_scheduler, run_scheduler

# 清除代理设置，避免网络连接问题
if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

if __name__ == '__main__':
    # 初始化配置
    init_cfg = global_cfg.clone()
    args = parse_args()
    
    # 从配置文件加载配置
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    
    # 解析命令行参数并合并到配置中
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    # 为超参数优化更新实验名称
    if init_cfg.expname == '':
        from federatedscope.autotune.utils import generate_hpo_exp_name
        init_cfg.expname = generate_hpo_exp_name(init_cfg)

    # 更新日志配置并设置随机种子
    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # 加载客户端配置文件
    if args.client_cfg_file:
        client_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        # client_cfgs.set_new_allowed(True)
        client_cfgs.merge_from_list(client_cfg_opt)
    else:
        client_cfgs = None

    # 获取并运行超参数优化调度器
    scheduler = get_scheduler(init_cfg, client_cfgs)
    run_scheduler(scheduler, init_cfg, client_cfgs)
