# 导入必要的库
import torch
import torch.nn as nn
from collections import OrderedDict


def enable_adapter(model, package, adapter, **kwargs):
    """
    为给定的模型和包启用适配器

    Args:
        model: 来自 HuggingFace Transformers 库的预训练模型
        package: 提供适配器的包名称字符串。目前仅支持 'peft' 和 'adapterhub'
        adapter: 要启用的适配器名称字符串。可用的适配器取决于包
        **kwargs: 传递给适配器配置的额外关键字参数

    Returns:
        启用了适配器的模型对象

    Raises:
        NotImplementedError: 如果包或适配器不受支持
    """
    adapter = adapter.lower()
    if package == 'peft':
        """
        PEFT: https://github.com/huggingface/peft
        支持的方法:
            LoRA - 低秩适应
            Prefix Tuning - 前缀调优
            P-Tuning - P调优
            Prompt Tuning - 提示调优
            AdaLoRA - 自适应LoRA
            VeRA - 向量相对适应
        """
        from peft import get_peft_model, TaskType
        if adapter == 'lora':
            # LoRA (Low-Rank Adaptation) 配置
            from peft import LoraConfig
            peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'vera':
            # VeRA (Vector-based Random Matrix Adaptation) 配置
            from peft import VeraConfig
            peft_config = VeraConfig(task_type=TaskType.SEQ_CLS, **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'prefix':
            # 前缀调优配置
            from peft import PrefixTuningConfig
            peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_CLS,
                                             **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'prompt':
            # 提示调优配置
            from peft import PromptTuningConfig
            peft_config = PromptTuningConfig(task_type=TaskType.SEQ_CLS,
                                             **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'p-tuning':
            # P-调优配置
            from peft import PromptEncoderConfig
            peft_config = PromptEncoderConfig(task_type=TaskType.SEQ_CLS,
                                              **kwargs)
            model = get_peft_model(model, peft_config)
        else:
            raise NotImplementedError
        # 打印可训练参数信息
        model.print_trainable_parameters()

    elif package == 'adapterhub':
        """
        AdapterHub: https://docs.adapterhub.ml/model_overview.html
        支持的方法:
            Bottleneck Adapters - 瓶颈适配器
            Prefix Tuning - 前缀调优
            LoRA - 低秩适应
            Compacter - 压缩器
            Adapter Fusion - 适配器融合
            Invertible Adapters - 可逆适配器
            Parallel block - 并行块
        """
        # TODO: 支持 adapterhub 后，将以下参数移至 yaml 文件以方便用户使用
        if adapter == 'lora':
            # LoRA 适配器配置
            from transformers.adapters import LoRAConfig

            config = LoRAConfig(r=8, alpha=16)
            model.add_adapter("lora_adapter", config=config)
            model.train_adapter(['lora_adapter'])
        elif adapter == 'bottleneck':
            # 瓶颈适配器配置
            from transformers.adapters import AdapterConfig

            config = AdapterConfig(mh_adapter=True,
                                   output_adapter=True,
                                   reduction_factor=16,
                                   non_linearity="relu")
            model.add_adapter("bottleneck_adapter", config=config)
            model.train_adapter(['bottleneck_adapter'])
        elif adapter == 'lang':
            # 语言适配器配置
            from transformers.adapters import PfeifferInvConfig

            config = PfeifferInvConfig()
            model.add_adapter("lang_adapter", config=config)
            model.train_adapter(['lang_adapter'])
        elif adapter == 'prefix':
            # 前缀调优配置
            from transformers.adapters import PrefixTuningConfig

            config = PrefixTuningConfig(flat=False, prefix_length=30)
            model.add_adapter("prefix_tuning", config=config)
            model.train_adapter(['prefix_tuning'])
        elif adapter == 'compacter':
            # Compacter 适配器配置
            from transformers.adapters import CompacterConfig

            config = CompacterConfig()
            model.add_adapter("dummy", config=config)
            model.train_adapter(['dummy'])
        elif adapter == 'ia_3':
            # IA³ 适配器配置
            from transformers.adapters import IA3Config

            config = IA3Config()
            model.add_adapter("ia3_adapter", config=config)
            model.train_adapter(['ia3_adapter'])
        elif adapter == 'union':
            # 联合适配器配置
            from transformers.adapters import AdapterConfig, ConfigUnion

            # TODO: 在配置文件中配置这些参数
            config = ConfigUnion(
                AdapterConfig(mh_adapter=True,
                              output_adapter=False,
                              reduction_factor=16,
                              non_linearity="relu"),
                AdapterConfig(mh_adapter=False,
                              output_adapter=True,
                              reduction_factor=2,
                              non_linearity="relu"),
            )
            model.add_adapter("union_adapter", config=config)
            model.train_adapter(['union_adapter'])
        elif adapter == 'mam':
            # MAM (Mix-and-Match) 适配器配置
            from transformers.adapters import \
                ConfigUnion, ParallelConfig, PrefixTuningConfig

            config = ConfigUnion(
                PrefixTuningConfig(bottleneck_size=800),
                ParallelConfig(),
            )
            model.add_adapter("mam_adapter", config=config)
            model.train_adapter(['mam_adapter'])
        else:
            raise NameError(
                f"在 {package} 中没有名为 {adapter} 的适配器")
    else:
        raise NotImplementedError
    return model


class AdapterModel(nn.Module):
    """
    可以使用适配器进行微调的模型包装类

    该类继承自 torch.nn.Module，实现了一个模型包装器，可以选择性地使用适配器
    进行微调。适配器是可以插入到预训练模型层之间的小模块，在特定任务上进行
    训练，同时保持原始参数冻结。该类可以使用不同的适配器包和方法，如 PEFT
    和 LoRA。它还提供了保存和加载模型状态字典以及使用模型生成文本的方法。

    Attributes:
        model: 表示原始或适配模型的 torch.nn.Module 对象

    """
    def __init__(self, model, use_adapter=False, *args, **kwargs):
        """
        使用给定的模型和参数初始化包装器

        Args:
            model: 表示原始模型的 torch.nn.Module 对象
            use_adapter: 布尔值，指示是否使用适配器进行微调。默认为 False
            *args: 传递给适配器包或方法的额外位置参数
            **kwargs: 传递给适配器包或方法的额外关键字参数。
                     可能包括 adapter_package、adapter_method 等
        """
        super().__init__()

        self.model = None
        if use_adapter:
            # 从参数中提取适配器包和方法
            adapter_package = kwargs.pop('adapter_package', 'peft')
            adapter_method = kwargs.pop('adapter_method', 'lora')

            # 启用适配器
            self.model = enable_adapter(model, adapter_package, adapter_method,
                                        **kwargs)
        else:
            # 不使用适配器，直接使用原始模型
            self.model = model

    def forward(self, *args, **kwargs):
        """
        调用包装模型的前向传播方法

        Args:
            *args: 传递给模型前向传播方法的位置参数
            **kwargs: 传递给模型前向传播方法的关键字参数

        Returns:
            模型前向传播方法的输出
        """
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        调用包装模型的生成方法

        Args:
            *args: 传递给模型生成方法的位置参数
            **kwargs: 传递给模型生成方法的关键字参数

        Returns:
            模型生成方法的输出
        """
        try:
            res = self.model.generate(*args, **kwargs)
        except RuntimeError as e:
            # 在 HELM 评估时，半精度可能会导致 RuntimeError
            # 以下代码解决了这个问题
            if 'do_sample' in kwargs.keys():
                del kwargs['do_sample']
                res = self.model.generate(*args, **kwargs)
            else:
                raise RuntimeError(e)
        return res

    def state_dict(self, return_trainable=True, *args, **kwargs):
        """
        返回包装模型的状态字典

        Args:
            return_trainable: 布尔值，指示是否仅返回模型的可训练参数。默认为 True
            *args: 传递给模型 state_dict 方法的额外位置参数
            **kwargs: 传递给模型 state_dict 方法的额外关键字参数

        Returns:
            包含模型状态字典的字典。如果 return_trainable 为 True，
            则仅包含需要梯度的参数。否则，包含所有参数。
        """
        if return_trainable:
            # 返回仅可训练的参数
            return self.get_trainable_state_dict()
        else:
            # 返回所有参数
            return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=False):
        """
        将状态字典加载到包装模型中

        Args:
            state_dict: 包含要加载到模型中的状态字典的字典
            strict: 布尔值，指示是否严格强制 state_dict 中的键
                   与此模块的 state_dict() 函数返回的键匹配。默认为 False
        """
        return self.model.load_state_dict(state_dict, strict=False)

    def get_trainable_state_dict(self):
        """
        仅返回包装模型的可训练参数

        此方法可用于获取仅需要梯度的参数，如适配器或任务特定层。

        Returns:
            包含模型可训练参数状态字典的字典
        """
        # 收集所有需要梯度的参数名称
        grad_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_params.append(name)
        
        # 获取完整的模型状态字典
        model_state_dict = self.model.state_dict()
        
        # 构建仅包含可训练参数的新状态字典
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            if k in grad_params:
                new_state_dict[k] = v
        return new_state_dict

    def save_model(self, path, state=0):
        """
        将模型状态字典和当前轮次保存到文件

        Args:
            path: 表示保存模型的文件路径的字符串
            state: 表示当前训练或评估轮次的整数。默认为 0

        """
        # 创建检查点字典，包含当前轮次和模型状态
        ckpt = {'cur_round': state, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    # TODO: 修复 `__getattr__` 方法
    # def __getattr__(self, item):
    #     return getattr(self.model, item)
