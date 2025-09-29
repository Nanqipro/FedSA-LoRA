# 导入 Transformers 库的分词器和数据集加载工具
from transformers import AutoTokenizer
from datasets import load_dataset

# GLUE 任务到输入键的映射
# 定义每个 GLUE 任务的输入文本字段名称
task_to_keys = {
    "cola": ("sentence", None),  # CoLA: 语言可接受性语料库，单句任务
    "mnli": ("premise", "hypothesis"),  # MNLI: 多体裁自然语言推理，前提-假设对
    "mrpc": ("sentence1", "sentence2"),  # MRPC: 微软研究释义语料库，句子对
    "qnli": ("question", "sentence"),  # QNLI: 问答自然语言推理，问题-句子对
    "qqp": ("question1", "question2"),  # QQP: Quora 问题对，问题对
    "rte": ("sentence1", "sentence2"),  # RTE: 识别文本蕴含，句子对
    "sst2": ("sentence", None),  # SST-2: 斯坦福情感树库，单句任务
    "stsb": ("sentence1", "sentence2"),  # STS-B: 语义文本相似性基准，句子对
    "wnli": ("sentence1", "sentence2"),  # WNLI: Winograd 自然语言推理，句子对
}


def load_glue_dataset(config=None, **kwargs):
    """
    加载 GLUE 数据集并进行预处理
    
    Args:
        config: 配置对象，包含模型和数据相关配置
        **kwargs: 其他关键字参数
        
    Returns:
        tuple: (数据集元组, 更新后的配置对象)
            数据集元组包含 (训练集, 验证集, 测试集)
    """
    # 从配置中提取模型名称和任务名称
    model_name, _ = config.model.type.split('@')
    task_name, _ = config.data.type.split('@')
    
    # 下载数据集
    datasets = load_dataset("glue", task_name, cache_dir=config.data.root)
    
    # 处理标签信息
    # 判断是否为回归任务（STS-B 是唯一的回归任务）
    is_regression = task_name == "stsb"
    if not is_regression:
        # 分类任务：获取标签列表和数量
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
        config.data.label_list = label_list  # 更新配置对象中的标签列表
    else:
        # 回归任务：标签数量为1
        num_labels = 1
    config.data.num_labels = num_labels  # 更新配置对象中的标签数量
    
    # 数据集预处理
    # 获取当前任务的输入字段键名
    sentence1_key, sentence2_key = task_to_keys[task_name]
    
    # 加载分词器
    # 获取缓存目录，如果未配置则使用默认值
    cache_dir = getattr(getattr(getattr(config, 'llm', None), 'cache', None), 'model', None)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    
    def preprocess_function(examples):
        """
        数据预处理函数：对文本进行分词和编码
        
        Args:
            examples: 批量样本数据
            
        Returns:
            dict: 包含 input_ids, attention_mask 等的字典
        """
        # 根据任务类型构建分词参数
        # 单句任务只有一个输入，句子对任务有两个输入
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        # 使用分词器进行编码，设置最大长度和填充策略
        result = tokenizer(*args, padding='max_length', max_length=config.llm.tok_len, truncation=True)
        return result
    
    # 对数据集应用预处理函数
    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=True)
    # 设置数据格式为 PyTorch 张量
    datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # 构建数据集分割
    train_dataset = datasets["train"]
    
    # MNLI 任务有匹配和不匹配两种验证集
    if task_name == "mnli":
        # 根据配置选择匹配或不匹配的验证集
        eval_dataset = datasets["validation_matched" if config.data.matched else "validation_mismatched"]
        # test_dataset = datasets["test_matched" if config.data.matched else "test_mismatched"]
    else:
        # 其他任务使用标准验证集
        eval_dataset = datasets["validation"]
        # test_dataset = datasets["test"]
    
    # 返回数据集元组：(训练集, 验证集, 测试集)
    # 注意：测试集暂时为空列表，因为 GLUE 测试集标签不公开
    dataset = (train_dataset, eval_dataset, [])

    return dataset, config
