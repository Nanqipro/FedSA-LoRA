"""自然语言处理训练器

该模块提供了专门用于自然语言处理任务的训练器实现。
主要功能包括：
- 继承自通用PyTorch训练器，专门优化文本数据处理
- 支持字典格式和张量格式的输入数据
- 自动处理设备移动和批次前向传播
- 适用于各种NLP任务（文本分类、序列标注等）

特性：
- 灵活的输入格式支持
- 自动标签维度处理
- 设备自适应
- 与联邦学习框架无缝集成
"""

from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.utils import move_to


class NLPTrainer(GeneralTorchTrainer):
    """自然语言处理训练器
    
    专门用于处理文本数据的训练器，继承自GeneralTorchTrainer。
    支持多种输入格式和NLP任务类型。
    
    特性:
        - 支持字典格式输入（如BERT模型的input_ids, attention_mask等）
        - 支持张量格式输入（如传统RNN模型）
        - 自动处理标签维度
        - 设备自适应移动
    
    注意:
        该训练器主要重写了批次前向传播的钩子函数，以适应NLP模型的特殊输入格式。
    """
    def _hook_on_batch_forward(self, ctx):
        """批次前向传播钩子函数
        
        处理NLP模型的前向传播，支持多种输入格式。
        
        参数:
            ctx: 训练上下文对象，包含数据批次、模型、设备等信息
        
        注意:
            - 自动将数据移动到指定设备
            - 支持字典格式输入（Transformer模型）和张量格式输入（RNN模型）
            - 自动处理标签维度问题
        """
        # 将输入数据和标签移动到指定设备
        x, label = [move_to(_, ctx.device) for _ in ctx.data_batch]
        
        # 根据输入格式选择不同的前向传播方式
        if isinstance(x, dict):
            # 字典格式输入（如Transformer模型的input_ids, attention_mask等）
            pred = ctx.model(**x)[0]
        else:
            # 张量格式输入（如传统RNN模型）
            pred = ctx.model(x)
        
        # 处理标签维度：如果是标量，转换为1维张量
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        
        # 计算损失
        ctx.loss_batch = ctx.criterion(pred, label)
        # 保存真实标签和预测结果
        ctx.y_true = label
        ctx.y_prob = pred
        # 记录批次大小
        ctx.batch_size = len(label)


def call_nlp_trainer(trainer_type):
    """调用NLP训练器构建函数
    
    根据训练器类型返回相应的NLP训练器类。
    
    参数:
        trainer_type (str): 训练器类型，目前支持 'nlptrainer'
    
    返回:
        训练器类: 对应的NLP训练器类
    
    注意:
        目前仅支持 'nlptrainer' 类型，返回 NLPTrainer 类
    """
    if trainer_type == 'nlptrainer':
        # 返回NLP训练器类
        trainer_builder = NLPTrainer
        return trainer_builder


# 注册NLP训练器到全局训练器注册表
register_trainer('nlptrainer', call_nlp_trainer)
