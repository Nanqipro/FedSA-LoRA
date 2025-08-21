"""自然语言处理模型构建器

该模块提供了构建自然语言处理模型的功能，支持循环神经网络（RNN）和Transformer模型。
主要功能包括：
- 构建LSTM等循环神经网络模型
- 构建基于transformers库的预训练Transformer模型
- 支持多种NLP任务（文本分类、问答、序列标注等）

支持的模型类型：
- RNN系列：LSTM
- Transformer系列：BERT、RoBERTa、GPT等预训练模型

支持的任务类型：
- PreTraining：预训练任务
- QuestionAnswering：问答任务
- SequenceClassification：序列分类任务
- TokenClassification：标记分类任务
- WithLMHead：语言模型任务
- Auto：自动任务
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


def get_rnn(model_config, input_shape):
    """构建循环神经网络模型
    
    根据配置参数构建RNN模型，目前支持LSTM。
    
    参数:
        model_config: 模型配置对象，包含模型类型、隐藏层大小等参数
        input_shape: 输入数据的形状，格式为 (batch_size, seq_len, hidden) 或 (seq_len, hidden)
    
    返回:
        构建好的RNN模型实例
    
    异常:
        ValueError: 当指定的模型类型不支持时抛出
    
    注意:
        - 目前仅支持LSTM模型
        - 输入形状的最后一维将作为输入通道数
    """
    from federatedscope.nlp.model.rnn import LSTM
    # 检查任务类型和输入形状
    # input_shape: (batch_size, seq_len, hidden) 或 (seq_len, hidden)
    if model_config.type == 'lstm':
        # 构建LSTM模型
        model = LSTM(
            # 输入通道数：使用配置中的值或从输入形状推断
            in_channels=input_shape[-2]
            if not model_config.in_channels else model_config.in_channels,
            hidden=model_config.hidden,  # 隐藏层大小
            out_channels=model_config.out_channels,  # 输出通道数
            embed_size=model_config.embed_size,  # 嵌入层大小
            dropout=model_config.dropout)  # Dropout概率
    else:
        # 不支持的模型类型
        raise ValueError(f'No model named {model_config.type}!')

    return model


def get_transformer(model_config, input_shape):
    """构建Transformer模型
    
    根据配置参数构建基于transformers库的预训练Transformer模型。
    支持多种NLP任务和预训练模型。
    
    参数:
        model_config: 模型配置对象，包含任务类型、模型路径等参数
        input_shape: 输入数据的形状（此参数在Transformer模型中通常不使用）
    
    返回:
        构建好的Transformer模型实例
    
    异常:
        AssertionError: 当指定的任务类型不支持时抛出
    
    注意:
        - 模型路径格式为 'model_name@version'，函数会自动解析路径
        - 支持的任务类型必须在预定义的字典中
        - 模型会根据out_channels参数设置标签数量
    """
    from transformers import AutoModelForPreTraining, \
        AutoModelForQuestionAnswering, AutoModelForSequenceClassification, \
        AutoModelForTokenClassification, AutoModelWithLMHead, AutoModel

    # 定义任务类型到模型类的映射字典
    model_func_dict = {
        'PreTraining'.lower(): AutoModelForPreTraining,  # 预训练任务
        'QuestionAnswering'.lower(): AutoModelForQuestionAnswering,  # 问答任务
        'SequenceClassification'.lower(): AutoModelForSequenceClassification,  # 序列分类任务
        'TokenClassification'.lower(): AutoModelForTokenClassification,  # 标记分类任务
        'WithLMHead'.lower(): AutoModelWithLMHead,  # 语言模型任务
        'Auto'.lower(): AutoModel  # 自动任务
    }
    # 验证任务类型是否支持
    assert model_config.task.lower(
    ) in model_func_dict, f'model_config.task should be in' \
                          f' {model_func_dict.keys()} ' \
                          f'when using pre_trained transformer model '
    # 解析模型路径（格式：model_name@version）
    path, _ = model_config.type.split('@')
    # 根据任务类型加载预训练模型
    model = model_func_dict[model_config.task.lower()].from_pretrained(
        path, num_labels=model_config.out_channels)  # 设置输出标签数量

    return model
