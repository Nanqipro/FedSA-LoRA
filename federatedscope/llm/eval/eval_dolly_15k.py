"""Dolly-15K 数据集评估模块。

该模块提供了用于评估 Dolly-15K 数据集上模型性能的评估函数。
Dolly-15K 是一个包含 15,000 个高质量人工生成指令-响应对的数据集。

主要功能:
    - ROUGE-L 分数计算：用于评估生成文本的质量
    - 准确率计算：用于分类任务的性能评估

评估指标:
    - ROUGE-L F1 分数：衡量生成文本与参考文本的重叠程度
    - 准确率：预测正确的样本比例
"""

from rouge import Rouge
import numpy as np

rouge = Rouge()


def rouge_score(hyp_ids, ref_ids, tokenizer):
    """
    计算假设文本和参考文本之间的 ROUGE-L F1 分数。
    
    ROUGE-L 是基于最长公共子序列（LCS）的评估指标，用于衡量
    生成文本与参考文本之间的相似性。F1 分数结合了精确率和召回率。
    
    Args:
        hyp_ids: 假设文本的标记 ID 列表
        ref_ids: 参考文本的标记 ID 列表  
        tokenizer: 用于解码标记 ID 的分词器
        
    Returns:
        float: ROUGE-L F1 分数，范围 [0, 1]，值越高表示相似性越好
               如果解码后的假设文本为空或计算出错，返回 0.0
    """
    # 将标记 ID 解码为文本
    hyps = [tokenizer.decode(hyp_ids, skip_special_tokens=True)]
    if len(hyps[0]) == 0:
        return 0.0
    refs = [tokenizer.decode(ref_ids, skip_special_tokens=True)]
    try:
        # 计算 ROUGE-L F1 分数
        rouge_score = rouge.get_scores(hyps, refs)[0]['rouge-l']['f']
    except ValueError:
        return 0.0
    return rouge_score


def acc_score(preds, labels):
    """
    计算预测结果的准确率。
    
    准确率是分类任务中最常用的评估指标，表示预测正确的样本
    占总样本数的比例。
    
    Args:
        preds: 预测结果列表或数组
        labels: 真实标签列表或数组
        
    Returns:
        float: 准确率，范围 [0, 1]，值越高表示预测性能越好
    """
    preds = np.array(preds)
    labels = np.array(labels)
    return np.sum(preds == labels) / float(len(labels))