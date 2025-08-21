# 导入必要的库
import torch
from federatedscope.glue.model.adapter_builder import AdapterModel


def get_model_from_huggingface(model_name, config):
    """
    从 Hugging Face 加载序列分类模型
    
    Args:
        model_name (str): 模型名称
        config: 配置对象
        
    Returns:
        AutoModelForSequenceClassification: 预训练的序列分类模型
    """
    from transformers import AutoModelForSequenceClassification

    # 构建模型加载参数
    kwargs = {}
    # 设置模型缓存目录
    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model
    
    # 为 GLUE 任务设置标签数量
    kwargs['num_labels'] = config.data.num_labels

    return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)


def get_llm(config):
    """
    获取大语言模型并应用适配器
    
    Args:
        config: 配置对象，包含模型类型、适配器配置等
        
    Returns:
        AdapterModel: 包装了适配器的模型
    """
    # 解析模型名称和模型源
    model_name, model_hub = config.model.type.split('@')
    
    # 根据模型源加载模型
    if model_hub == 'huggingface_llm':
        model = get_model_from_huggingface(model_name=model_name,
                                           config=config)
    else:
        raise NotImplementedError(f'Not support LLM {model_name} in'
                                  f' {model_hub}.')

    # 获取适配器参数
    args = config.llm.adapter.args[0] if len(
        config.llm.adapter.args[0]) > 0 else {}
    # 使用适配器包装模型
    model = AdapterModel(model, use_adapter=config.llm.adapter.use, **args)
    
    # FFA-LoRA & FFA-VeRA: 冻结 A 矩阵
    # 在联邦学习中，可以选择冻结 LoRA 的 A 矩阵或 VeRA 的 lambda_d 参数
    if config.federate.freeze_A:
        for name, param in model.named_parameters():
            if "lora_A" in name or "vera_lambda_d" in name:
                param.requires_grad = False
    
    # 保存初始 LoRA 参数，用于本地训练
    if config.federate.method == "local":
        # 根据适配器类型选择要保存的参数
        if config.llm.adapter.args[0].get('adapter_method', '') == "vera":
            # VeRA 适配器：保存所有包含 'vera' 的参数
            initial_lora_params = {name: param.clone() for name, param in model.named_parameters() if 'vera' in name}
        else:
            # LoRA 适配器：保存所有包含 'lora' 的参数
            initial_lora_params = {name: param.clone() for name, param in model.named_parameters() if 'lora' in name}
        # 将初始参数保存到文件
        torch.save(initial_lora_params, config.federate.save_to + '.init')
    
    return model
