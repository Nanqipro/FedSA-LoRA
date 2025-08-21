"""LLM 适配器构建器模块。

该模块提供了用于为大型语言模型（LLM）添加和管理适配器的功能。
适配器是可以插入预训练模型层之间的小模块，在特定任务上训练，同时保持原始参数冻结。

主要功能:
    - 支持多种适配器包（PEFT、AdapterHub）
    - 支持多种适配器方法（LoRA、前缀调优、提示调优等）
    - 提供适配器模型包装类
    - 支持模型状态字典的保存和加载
    - 支持仅保存可训练参数

支持的适配器方法:
    PEFT 包:
        - LoRA: 低秩适应
        - Prefix Tuning: 前缀调优
        - Prompt Tuning: 提示调优
        - P-Tuning: P调优
        - AdaLoRA: 自适应LoRA
    
    AdapterHub 包:
        - Bottleneck Adapters: 瓶颈适配器
        - LoRA: 低秩适应
        - Compacter: 压缩适配器
        - IA3: (IA)³ 适配器
        - Union: 联合适配器
        - MAM: Mix-and-Match 适配器

主要类:
    - AdapterModel: 适配器模型包装类

主要函数:
    - enable_adapter: 为模型启用适配器
"""

# 导入必要的库
import torch
import torch.nn as nn
from collections import OrderedDict


def enable_adapter(model, package, adapter, **kwargs):
    """
    为给定的模型和包启用适配器。
    Enables an adapter for a given model and package.

    Args:
        model: 来自 HuggingFace Transformers 库的预训练模型 / A pre-trained model from HuggingFace Transformers library.
        package: 提供适配器的包名称字符串，目前仅支持 'peft' 和 'adapterhub' / A string indicating the name of the package that provides the adapter.
        adapter: 要启用的适配器名称字符串，可用适配器取决于包 / A string indicating the name of the adapter to enable.
        **kwargs: 传递给适配器配置的额外关键字参数 / Additional keyword arguments that are passed to the adapter configuration.

    Returns:
        启用了适配器的模型对象 / A model object that has the adapter enabled.

    Raises:
        NotImplementedError: 如果包或适配器不受支持 / If the package or the adapter is not supported.
    """
    adapter = adapter.lower()  # 将适配器名称转换为小写
    if package == 'peft':
        """
        PEFT: https://github.com/huggingface/peft
        支持的方法 / Support methods:
            LoRA - 低秩适应
            Prefix Tuning - 前缀调优
            P-Tuning - P调优
            Prompt Tuning - 提示调优
            AdaLoRA - 自适应LoRA
        """
        from peft import get_peft_model, TaskType
        if adapter == 'lora':
            # 配置 LoRA 适配器
            from peft import LoraConfig
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'prefix':
            # 配置前缀调优适配器
            from peft import PrefixTuningConfig
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,
                                             **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'prompt':
            # 配置提示调优适配器
            from peft import PromptTuningConfig
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM,
                                             **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'p-tuning':
            # 配置 P-Tuning 适配器
            from peft import PromptEncoderConfig
            peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM,
                                              **kwargs)
            model = get_peft_model(model, peft_config)
        else:
            raise NotImplementedError
        model.print_trainable_parameters()  # 打印可训练参数信息

    elif package == 'adapterhub':
        """
        AdapterHub: https://docs.adapterhub.ml/model_overview.html
        支持的方法 / Support methods:
            Bottleneck Adapters - 瓶颈适配器
            Prefix Tuning - 前缀调优
            LoRA - 低秩适应
            Compacter - 压缩适配器
            Adapter Fusion - 适配器融合
            Invertible Adapters - 可逆适配器
            Parallel block - 并行块
        """
        # TODO: 支持 adapterhub 后，将以下参数移至 yaml 文件以便用户使用
        #   parameters in yaml file for users' convenient
        if adapter == 'lora':
            # 配置 LoRA 适配器
            from transformers.adapters import LoRAConfig

            config = LoRAConfig(r=8, alpha=16)
            model.add_adapter("lora_adapter", config=config)
            model.train_adapter(['lora_adapter'])
        elif adapter == 'bottleneck':
            # 配置瓶颈适配器
            from transformers.adapters import AdapterConfig

            config = AdapterConfig(mh_adapter=True,
                                   output_adapter=True,
                                   reduction_factor=16,
                                   non_linearity="relu")
            model.add_adapter("bottleneck_adapter", config=config)
            model.train_adapter(['bottleneck_adapter'])
        elif adapter == 'lang':
            # 配置语言适配器
            from transformers.adapters import PfeifferInvConfig

            config = PfeifferInvConfig()
            model.add_adapter("lang_adapter", config=config)
            model.train_adapter(['lang_adapter'])
        elif adapter == 'prefix':
            # 配置前缀调优适配器
            from transformers.adapters import PrefixTuningConfig

            config = PrefixTuningConfig(flat=False, prefix_length=30)
            model.add_adapter("prefix_tuning", config=config)
            model.train_adapter(['prefix_tuning'])
        elif adapter == 'compacter':
            # 配置压缩适配器
            from transformers.adapters import CompacterConfig

            config = CompacterConfig()
            model.add_adapter("dummy", config=config)
            model.train_adapter(['dummy'])
        elif adapter == 'ia_3':
            # 配置 IA3 适配器
            from transformers.adapters import IA3Config

            config = IA3Config()
            model.add_adapter("ia3_adapter", config=config)
            model.train_adapter(['ia3_adapter'])
        elif adapter == 'union':
            # 配置联合适配器
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
            # 配置 MAM (Mix-and-Match) 适配器
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
                f"There is no adapter named {adapter} in {package}")
    else:
        raise NotImplementedError
    return model  # 返回启用了适配器的模型


class AdapterModel(nn.Module):
    """
    可以使用适配器进行微调的模型包装类。
    A wrapper class for a model that can use adapters for fine-tuning.

    该类继承自 torch.nn.Module，实现了一个可选择性使用适配器进行微调的模型包装器。
    适配器是可以插入预训练模型层之间的小模块，在特定任务上训练，同时保持原始参数冻结。
    该类可以使用不同的适配器包和方法，如 PEFT 和 LoRA。它还提供了保存和加载模型状态字典以及使用模型生成文本的方法。
    
    This class inherits from torch.nn.Module and implements a wrapper for a
    model that can optionally use adapters for fine-tuning. Adapters are small
    modules that can be inserted between the layers of a pretrained model and
    trained on a specific task, while keeping the original parameters frozen.
    This class can use different adapter packages and methods, such as PEFT
    and LoRA. It also provides methods for saving and loading the model state
    dict, as well as generating text using the model.

    Attributes:
        model: 表示原始或适配模型的 torch.nn.Module 对象 / A torch.nn.Module object that represents the original or adapted model.

    """
    def __init__(self, model, use_adapter=False, *args, **kwargs):
        """
        使用给定的模型和参数初始化包装器。
        Initializes the wrapper with the given model and arguments.

        Args:
            model: 表示原始模型的 torch.nn.Module 对象 / A torch.nn.Module object that represents the original model.
            use_adapter: 是否使用适配器进行微调的布尔值，默认为 False / A boolean indicating whether to use adapters for fine-tuning.
            *args: 传递给适配器包或方法的额外位置参数 / Additional positional arguments to pass to the adapter package or method.
            **kwargs: 传递给适配器包或方法的额外关键字参数，可能包括 adapter_package、adapter_method 等 / Additional keyword arguments.
        """
        super().__init__()

        self.model = None
        if use_adapter:
            # 从参数中获取适配器包和方法，默认使用 PEFT 和 LoRA
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
        调用包装模型的前向传播方法。
        Calls the forward method of the wrapped model.

        Args:
            *args: 传递给模型前向传播方法的位置参数 / Positional arguments to pass to the model's forward method.
            **kwargs: 传递给模型前向传播方法的关键字参数 / Keyword arguments to pass to the model's forward method.

        Returns:
            模型前向传播方法的输出 / The output of the model's forward method.
        """
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        调用包装模型的生成方法。
        Calls the generate method of the wrapped model.

        Args:
            *args: 传递给模型生成方法的位置参数 / Positional arguments to pass to the model's generate method.
            **kwargs: 传递给模型生成方法的关键字参数 / Keyword arguments to pass to the model's generate method.

        Returns:
            模型生成方法的输出 / The output of the model's generate method.
        """
        try:
            res = self.model.generate(*args, **kwargs)
        except RuntimeError as e:
            # 在 HELM 评估时，半精度会导致 RuntimeError，以下代码解决此问题
            # When does evaluation in HELM, half precision will cause RuntimeError, the following solves it
            if 'do_sample' in kwargs.keys():
                del kwargs['do_sample']
                res = self.model.generate(*args, **kwargs)
            else:
                raise RuntimeError(e)
        return res

    def state_dict(self, return_trainable=True, *args, **kwargs):
        """
        返回包装模型的状态字典。
        Returns the state dict of the wrapped model.

        Args:
            return_trainable: 是否仅返回模型的可训练参数的布尔值，默认为 True / A boolean indicating whether to return only the trainable parameters.
            *args: 传递给模型 state_dict 方法的额外位置参数 / Additional positional arguments to pass to the model's state_dict method.
            **kwargs: 传递给模型 state_dict 方法的额外关键字参数 / Additional keyword arguments to pass to the model's state_dict method.

        Returns:
            包含模型状态字典的字典。如果 return_trainable 为 True，则仅包含需要梯度的参数，否则包含所有参数 / A dictionary containing the state dict.
        """
        if return_trainable:
            return self.get_trainable_state_dict()
        else:
            return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=False):
        """
        将状态字典加载到包装模型中。
        Loads the state dict into the wrapped model.

        Args:
            state_dict: 包含要加载到模型中的状态字典的字典 / A dictionary containing the state dict to load into the model.
            strict: 是否严格强制 state_dict 中的键与此模块的 state_dict() 函数返回的键匹配的布尔值，默认为 False / A boolean indicating whether to strictly enforce key matching.
        """
        return self.model.load_state_dict(state_dict, strict=False)

    def get_trainable_state_dict(self):
        """
        仅返回包装模型的可训练参数。
        Returns only the trainable parameters of the wrapped model.

        此方法可用于获取仅需要梯度的参数，如适配器或任务特定层。
        This method can be used to get only the parameters that require grad, such as adapters or task-specific layers.

        Returns:
            包含模型可训练参数状态字典的字典 / A dictionary containing the state dict of the trainable parameters of the model.
        """
        grad_params = []  # 存储需要梯度的参数名称
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_params.append(name)
        model_state_dict = self.model.state_dict()  # 获取完整的状态字典
        new_state_dict = OrderedDict()  # 创建新的有序字典
        for k, v in model_state_dict.items():
            if k in grad_params:
                new_state_dict[k] = v  # 仅保留可训练参数
        return new_state_dict

    def save_model(self, path, state=0):
        """
        将模型状态字典和当前轮次保存到文件。
        Saves the model state dict and the current round to a file.

        Args:
            path: 表示保存模型的文件路径的字符串 / A string representing the file path to save the model to.
            state: 表示当前训练或评估轮次的整数，默认为 0 / An integer representing the current round of training or evaluation.

        """
        ckpt = {'cur_round': state, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    # TODO: 修复 `__getattr__` 方法 / Fix `__getattr__`
    # def __getattr__(self, item):
    #     return getattr(self.model, item)
