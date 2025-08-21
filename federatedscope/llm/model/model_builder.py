"""LLM 模型构建器模块。

该模块提供了用于构建和配置大型语言模型（LLM）的功能，支持从不同模型中心
（如 HuggingFace 和 ModelScope）加载预训练模型，并可选择性地添加适配器层。

主要功能:
    - 从 HuggingFace transformers 库加载因果语言模型
    - 从 ModelScope 模型库加载因果语言模型
    - 根据配置创建带有适配器的 LLM 模型
    - 支持词嵌入大小调整和新标记初始化
    - 支持 FFA-LoRA 的 LoRA A 矩阵冻结功能

主要函数:
    - get_model_from_huggingface: 从 HuggingFace 加载模型
    - get_model_from_modelscope: 从 ModelScope 加载模型
    - get_llm: 根据配置获取完整的 LLM 模型
"""

# 导入适配器模型构建器
from federatedscope.llm.model.adapter_builder import AdapterModel


def get_model_from_huggingface(model_name, config):
    """
    从 HuggingFace transformers 库加载因果语言模型。
    Load a causal language model from HuggingFace transformers library.

    Args:
        model_name (str): 要加载的预训练模型名称 / The name of the pre-trained model to load.
        config (Config): 包含模型参数的配置对象 / The configuration object that contains the model parameters.

    Returns:
        AutoModelForCausalLM: 因果语言模型对象 / A causal language model object.
    """
    from transformers import AutoModelForCausalLM

    kwargs = {}
    # 如果配置了模型缓存目录，则使用它
    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def get_model_from_modelscope(model_name, config):
    """
    从 ModelScope 模型库加载因果语言模型。
    Load a causal language model from ModelScope models library.

    Args:
        model_name (str): 要加载的预训练模型名称 / The name of the pre-trained model to load.
        config (Config): 包含模型参数的配置对象 / The configuration object that contains the model parameters.

    Returns:
        Model: 因果语言模型对象 / A causal language model object.
    """
    from modelscope import AutoModelForCausalLM

    kwargs = {}
    # 如果配置了模型缓存目录，则使用它
    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def get_llm(config):
    """
    根据配置获取因果语言模型。
    Get a causal language model based on the configuration.

    Args:
        config (Config): 包含模型参数的配置对象 / The configuration object that contains the model parameters.

    Returns:
        AdapterModel: 带有可选适配器层的因果语言模型对象 / A causal language model object with optional adapter layers.
    """
    from federatedscope.llm.dataloader import get_tokenizer

    model_config = config.model
    # 解析模型名称和模型中心
    model_name, model_hub = model_config.type.split('@')
    if model_hub == 'huggingface_llm':
        # 从 HuggingFace 加载模型
        model = get_model_from_huggingface(model_name=model_name,
                                           config=config)
    elif model_hub == 'modelscope_llm':
        # 从 ModelScope 加载模型
        model = get_model_from_modelscope(model_name=model_name, config=config)
    else:
        # 不支持的模型中心
        raise NotImplementedError(f'Not support LLM {model_name} in'
                                  f' {model_hub}.')

    # 根据设置调整 LLM 模型大小
    tokenizer, num_new_tokens = \
        get_tokenizer(model_name, config.data.root, config.llm.tok_len,
                      model_hub)
    # 调整模型的词嵌入大小以匹配分词器
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        # 如果添加了新的标记，需要初始化它们的嵌入
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        # 使用现有嵌入的平均值初始化新标记的嵌入
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    # 获取适配器参数并创建适配器模型
    args = config.llm.adapter.args[0] if len(
        config.llm.adapter.args[0]) > 0 else {}
    model = AdapterModel(model, use_adapter=config.llm.adapter.use, **args)
    
    # FFA-LoRA: 冻结 LoRA A 矩阵
    # FFA-LoRA: Freeze LoRA A matrices
    if config.federate.freeze_A:
        for name, param in model.named_parameters():
            if "lora_A" in name:
                param.requires_grad = False  # 冻结 LoRA A 矩阵参数
    
    return model
