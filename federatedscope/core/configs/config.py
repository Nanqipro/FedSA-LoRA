# 导入必要的模块
import copy  # 用于深拷贝对象
import logging  # 日志记录
import os  # 操作系统接口

from pathlib import Path  # 路径操作

# 导入 FederatedScope 相关模块
import federatedscope.register as register  # 注册机制
from federatedscope.core.configs.yacs_config import CfgNode, _merge_a_into_b, \
    Argument  # YACS 配置系统相关类

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


def set_help_info(cn_node, help_info_dict, prefix=""):
    """
    递归设置配置节点的帮助信息
    
    Args:
        cn_node: 配置节点
        help_info_dict: 帮助信息字典
        prefix: 配置项前缀
    """
    for k, v in cn_node.items():
        if isinstance(v, Argument) and k not in help_info_dict:
            # 如果是 Argument 类型且不在帮助信息字典中，添加其描述
            help_info_dict[prefix + k] = v.description
        elif isinstance(v, CN):
            # 如果是 CN 类型，递归处理子配置
            set_help_info(v,
                          help_info_dict,
                          prefix=f"{k}." if prefix == "" else f"{prefix}{k}.")


class CN(CfgNode):
    """
    基于 YACS (https://github.com/rbgirshick/yacs) 的扩展配置系统。
    
    采用两级树结构，包含多个内部字典式容器，支持简单的键值访问和管理。
    提供配置验证、帮助信息、合并等功能。
    """
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        """
        初始化配置节点
        
        Args:
            init_dict: 初始化字典
            key_list: 键列表
            new_allowed: 是否允许添加新键
        """
        init_dict = super().__init__(init_dict, key_list, new_allowed)
        self.__cfg_check_funcs__ = list()  # 用于检查配置值有效性的函数列表
        self.__help_info__ = dict()  # 构建帮助信息字典

        self.is_ready_for_run = False  # 标记此配置节点是否已检查其有效性、完整性并清理无用信息

        # 如果提供了初始化字典，提取帮助信息
        if init_dict:
            for k, v in init_dict.items():
                if isinstance(v, Argument):
                    self.__help_info__[k] = v.description
                elif isinstance(v, CN) and "help_info" in v:
                    for name, des in v.__help_info__.items():
                        self.__help_info__[name] = des

    def __getattr__(self, name):
        """
        获取属性值
        
        Args:
            name: 属性名
            
        Returns:
            属性值
            
        Raises:
            AttributeError: 当属性不存在时
        """
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __delattr__(self, name):
        """
        删除属性
        
        Args:
            name: 属性名
            
        Raises:
            AttributeError: 当属性不存在时
        """
        if name in self:
            del self[name]
        else:
            raise AttributeError(name)

    def clear_aux_info(self):
        """
        清除 CN 对象的所有辅助信息
        
        包括配置检查函数、帮助信息和运行就绪标志等。
        递归清除所有子配置节点的辅助信息。
        """
        if hasattr(self, "__cfg_check_funcs__"):
            delattr(self, "__cfg_check_funcs__")
        if hasattr(self, "__help_info__"):
            delattr(self, "__help_info__")
        if hasattr(self, "is_ready_for_run"):
            delattr(self, "is_ready_for_run")
        # 递归清除子配置节点的辅助信息
        for v in self.values():
            if isinstance(v, CN):
                v.clear_aux_info()

    def print_help(self, arg_name=""):
        """
        打印特定参数或所有参数的帮助信息
        
        Args:
            arg_name: 特定参数名称，如果为空则打印所有参数的帮助信息
        """
        if arg_name != "" and arg_name in self.__help_info__:
            # 打印特定参数的帮助信息
            print(f"  --{arg_name} \t {self.__help_info__[arg_name]}")
        else:
            # 打印所有参数的帮助信息
            for k, v in self.__help_info__.items():
                print(f"  --{k} \t {v}")

    def register_cfg_check_fun(self, cfg_check_fun):
        """
        注册一个检查配置节点的函数
        
        Args:
            cfg_check_fun: 用于验证配置正确性的函数
        """
        self.__cfg_check_funcs__.append(cfg_check_fun)

    def merge_from_file(self, cfg_filename, check_cfg=True):
        """
        从 YAML 文件、另一个配置实例或存储键值对的列表中加载配置
        
        Args:
            cfg_filename: YAML 文件名
            check_cfg: 是否启用配置检查 (assert_cfg())
        """
        # 备份当前的配置检查函数
        cfg_check_funcs = copy.copy(self.__cfg_check_funcs__)
        # 从文件中加载配置
        with open(cfg_filename, "r") as f:
            cfg = self.load_cfg(f)
        # 合并配置
        self.merge_from_other_cfg(cfg)
        # 恢复配置检查函数
        self.__cfg_check_funcs__.clear()
        self.__cfg_check_funcs__.extend(cfg_check_funcs)
        # 执行配置检查
        self.assert_cfg(check_cfg)
        # 设置帮助信息
        set_help_info(self, self.__help_info__)

    def merge_from_other_cfg(self, cfg_other, check_cfg=True):
        """
        从另一个配置实例中加载配置
        
        Args:
            cfg_other: 要合并的其他配置
            check_cfg: 是否启用配置检查 (assert_cfg())
        """
        # 备份当前的配置检查函数
        cfg_check_funcs = copy.copy(self.__cfg_check_funcs__)
        # 将其他配置合并到当前配置中
        _merge_a_into_b(cfg_other, self, self, [])
        # 恢复配置检查函数
        self.__cfg_check_funcs__.clear()
        self.__cfg_check_funcs__.extend(cfg_check_funcs)
        # 执行配置检查
        self.assert_cfg(check_cfg)
        # 设置帮助信息
        set_help_info(self, self.__help_info__)

    def merge_from_list(self, cfg_list, check_cfg=True):
        """
        从存储键值对的列表中加载配置
        
        修改了 yacs.config.py 中的 merge_from_list 方法，
        当 is_new_allowed() 返回 True 时允许添加新键
        
        Args:
            cfg_list: 配置名称和值的配对列表
            check_cfg: 是否启用配置检查 (assert_cfg())
        """
        # 备份当前的配置检查函数
        cfg_check_funcs = copy.copy(self.__cfg_check_funcs__)
        # 调用父类方法合并列表配置
        super().merge_from_list(cfg_list)
        # 恢复配置检查函数
        self.__cfg_check_funcs__.clear()
        self.__cfg_check_funcs__.extend(cfg_check_funcs)
        # 执行配置检查
        self.assert_cfg(check_cfg)
        # 设置帮助信息
        set_help_info(self, self.__help_info__)

    def assert_cfg(self, check_cfg=True):
        """
        检查配置实例的有效性
        
        Args:
            check_cfg: 是否启用检查
        """
        if check_cfg:
            # 执行所有注册的配置检查函数
            for check_func in self.__cfg_check_funcs__:
                check_func(self)

    def clean_unused_sub_cfgs(self):
        """
        清理未使用的二级配置节点
        
        删除那些 .use 属性为 False 的子配置节点中的所有属性（除了 use 属性本身）
        """
        for v in self.values():
            if isinstance(v, CfgNode) or isinstance(v, CN):
                # 处理子配置
                if hasattr(v, "use") and v.use is False:
                    # 删除未使用配置的所有属性（除了 use 属性）
                    for k in copy.deepcopy(v).keys():
                        if k == "use":
                            continue  # 保留 use 属性
                        else:
                            del v[k]  # 删除其他属性

    def check_required_args(self):
        """
        检查必需的参数
        
        递归检查所有配置项，对于标记为必需但值为 None 的参数发出警告
        """
        for k, v in self.items():
            if isinstance(v, CN):
                # 递归检查子配置节点
                v.check_required_args()
            if isinstance(v, Argument) and v.required and v.value is None:
                # 对未设置的必需参数发出警告
                logger.warning(f"You have not set the required argument {k}")

    def de_arguments(self):
        """
        清理 Argument 类包装的配置值
        
        某些配置值通过 Argument 类管理，此函数用于清理这些值，
        移除 Argument 类包装，使类型特定的方法能正确工作，
        例如对字符串配置使用 len(cfg.federate.method)
        """
        for k, v in copy.deepcopy(self).items():
            if isinstance(v, CN):
                # 递归处理子配置节点
                self[k].de_arguments()
            if isinstance(v, Argument):
                # 将 Argument 对象替换为其实际值
                self[k] = v.value

    def ready_for_run(self, check_cfg=True):
        """
        检查并清理配置的内部状态，为运行做准备
        
        Args:
            check_cfg: 是否启用配置检查 (assert_cfg())
        """
        # 执行配置检查
        self.assert_cfg(check_cfg)
        # 清理未使用的子配置
        self.clean_unused_sub_cfgs()
        # 检查必需参数
        self.check_required_args()
        # 清理 Argument 包装
        self.de_arguments()
        # 标记配置已准备就绪
        self.is_ready_for_run = True

    def freeze(self, inform=True, save=True, check_cfg=True):
        """
        冻结配置属性使其不可变，并保存配置
        
        (1) 使配置属性不可变
        (2) 如果 save==True，将冻结的配置保存到 self.outdir/config.yaml 以提高可重现性
        (3) 如果 self.wandb.use==True，更新冻结的配置
        
        Args:
            inform: 是否打印配置信息
            save: 是否保存配置到文件
            check_cfg: 是否启用配置检查
        """
        # 准备配置以供运行
        self.ready_for_run(check_cfg)
        # 调用父类的 freeze 方法
        super(CN, self).freeze()

        if save:  # 保存最终配置
            # 创建输出目录
            Path(self.outdir).mkdir(parents=True, exist_ok=True)
            # 保存配置到 YAML 文件
            with open(os.path.join(self.outdir, "config.yaml"),
                      'w') as outfile:
                from contextlib import redirect_stdout
                with redirect_stdout(outfile):
                    # 创建配置的深拷贝并清除辅助信息
                    tmp_cfg = copy.deepcopy(self)
                    tmp_cfg.clear_aux_info()
                    print(tmp_cfg.dump())
                if self.wandb.use:
                    # 更新冻结的配置到 wandb
                    try:
                        import wandb
                        import yaml
                        cfg_yaml = yaml.safe_load(tmp_cfg.dump())
                        wandb.config.update(cfg_yaml, allow_val_change=True)
                    except ImportError:
                        logger.error(
                            "cfg.wandb.use=True but not install the wandb "
                            "package")
                        exit()

            if inform:
                # 打印使用的配置信息
                logger.info("the used configs are: \n" + str(tmp_cfg))


# 确保在设置全局配置之前注册所有子配置
# 导入核心配置模块列表
from federatedscope.core.configs import all_sub_configs

# 动态导入所有核心子配置模块
for sub_config in all_sub_configs:
    __import__("federatedscope.core.configs." + sub_config)

# 导入贡献配置模块列表
from federatedscope.contrib.configs import all_sub_configs_contrib

# 动态导入所有贡献子配置模块
for sub_config in all_sub_configs_contrib:
    __import__("federatedscope.contrib.configs." + sub_config)

# 全局配置对象
global_cfg = CN()


def init_global_cfg(cfg):
    """
    初始化全局配置的默认值
    
    注意事项：
    (1) 对于一个实验，只会使用部分参数，其余未使用的参数不会产生任何影响。
        因此可以随意在 graphgym.contrib.config 中注册任何参数
    (2) 我们支持多级配置，例如 cfg.dataset.name
    
    Args:
        cfg: 要初始化的配置对象
    """

    # ---------------------------------------------------------------------- #
    # 基本选项，第一级配置
    # ---------------------------------------------------------------------- #

    # 后端框架
    cfg.backend = 'torch'

    # 是否使用 GPU
    cfg.use_gpu = False

    # 是否检查消息处理器的完整性
    cfg.check_completeness = False

    # 是否打印详细的日志信息
    cfg.verbose = 1

    # 使用日志记录器打印时的小数位数
    cfg.print_decimal_digits = 6

    # 指定设备
    cfg.device = -1
    
    # 评估设备（用于评估 gsm8k）
    cfg.eval_device = -1   # added by me, for evaluating gsm8k

    # 随机种子
    cfg.seed = 0

    # 配置文件路径
    cfg.cfg_file = ''

    # 用于保存日志、实验配置、模型等的目录
    cfg.outdir = 'exp'
    cfg.expname = ''  # 详细的实验名称，用于区分不同的子实验
    cfg.expname_tag = ''  # 详细的实验标签，用于区分具有相同实验名称的不同子实验

    # 扩展用户自定义配置
    for func in register.config_dict.values():
        func(cfg)

    # 设置帮助信息
    set_help_info(cfg, cfg.__help_info__)


init_global_cfg(global_cfg)
