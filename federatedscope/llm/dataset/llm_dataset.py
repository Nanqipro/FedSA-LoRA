"""
大型语言模型数据集模块
LLM Dataset Module

部分代码片段借鉴自开源项目 stanford_alpaca
Some code snippets are borrowed from the open-sourced stanford_alpaca (
    https://github.com/tatsu-lab/stanford_alpaca)
"""

# 导入必要的库
import copy
import logging
import pandas as pd

from enum import Enum
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DefaultToken(Enum):
    """默认的特殊标记枚举类 / Default special tokens enumeration"""
    PAD_TOKEN = "[PAD]"  # 填充标记
    EOS_TOKEN = "</s>"   # 结束标记
    BOS_TOKEN = "<s>"    # 开始标记
    UNK_TOKEN = "<unk>"  # 未知标记
    IGNORE_INDEX = -100  # 忽略索引（用于损失计算）


# 提示模板字典 / Prompt template dictionary
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:"
        "\n{input}\n\n### Response:"),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"),
}


class LLMDataset(Dataset):
    """
    用于语言建模任务的数据集类。
    A dataset for language modeling tasks.

    该类继承自 torch.utils.data.Dataset，实现了一个可以加载和预处理语言建模数据的数据集。
    This class inherits from torch.utils.data.Dataset and implements a dataset that can load and preprocess data for language modeling.
    
    它接受数据字典列表、分词器和可选的提示模板作为输入，并创建输入 ID、标签和类别作为输出。
    It takes a list of data dictionaries, a tokenizer, and optional prompt templates as input, and creates input ids, labels, and categories as output.
    
    输入 ID 和标签根据分词器设置以及源和目标长度进行填充和掩码。类别使用 pandas.Categorical 编码为整数。
    The input ids and labels are padded and masked according to the tokenizer settings and the source and target lengths. The categories are encoded as integers using pandas.Categorical.

    Attributes:
        input_ids: 包含填充输入 ID 的 torch.LongTensor 对象列表，形状为 (max_length,) / A list of torch.LongTensor objects of shape (max_length,) containing the padded input ids.
        labels: 包含填充标签的 torch.LongTensor 对象列表，形状为 (max_length,) / A list of torch.LongTensor objects of shape (max_length,) containing the padded labels.
        categories: 表示类别代码的整数列表 / A list of integers representing the category codes.
        tokenizer: 可以编码和解码文本的 transformers.PreTrainedTokenizer 对象 / A transformers.PreTrainedTokenizer object that can encode and decode text.
    """
    def __init__(self,
                 list_data_dict,
                 tokenizer,
                 prompt_input=PROMPT_DICT["prompt_input"],
                 prompt_no_input=PROMPT_DICT["prompt_no_input"]):
        """
        使用给定参数初始化数据集。
        Initializes the dataset with the given arguments.

        Args:
            list_data_dict: 字典列表，每个字典包含输入、输出和可选的类别键值对（字符串形式）
                           A list of dictionaries, each containing input, output, and optionally category keys and values as strings.
            tokenizer: 可以编码和解码文本的 transformers.PreTrainedTokenizer 对象
                      A transformers.PreTrainedTokenizer object that can encode and decode text.
            prompt_input: 当数据字典中存在输入键时用于创建源文本的可选字符串模板
                         An optional string template for creating the source text when the input key is present in the data dictionary.
                         模板可以使用 {input}、{output} 和 {category} 作为相应值的占位符
                         The template can use {input}, {output}, and {category} as placeholders for the corresponding values.
            prompt_no_input: 当数据字典中不存在输入键时用于创建源文本的可选字符串模板
                            An optional string template for creating the source text when the input key is not present in the data dictionary.
                            模板可以使用 {output} 和 {category} 作为占位符
                            The template can use {output} and {category} as placeholders for the corresponding values.
        """
        super(LLMDataset, self).__init__()

        # 根据是否有输入选择相应的提示模板
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != ""
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        # 为每个目标添加结束标记
        targets = [
            f"{example['output']}{tokenizer.eos_token}"
            for example in list_data_dict
        ]

        # 预处理源文本和目标文本
        data_dict = self.preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        # 提取类别信息
        categories = [
            example['category'] if 'category' in example else None
            for example in list_data_dict
        ]
        # 将类别转换为数值编码
        df = pd.DataFrame(categories, columns=["category"])
        self.categories = list(pd.Categorical(df["category"]).codes)
        
        # 保存原始文本数据（用于 GSM8K 等数据集）
        # added by me, save original text data for gsm8k
        self.list_data_dict = list_data_dict  # for gsm8k

    def _tokenize_fn(self, strings, tokenizer):
        """
        使用给定的分词器对字符串列表进行分词。
        Tokenizes a list of strings using the given tokenizer.

        Args:
            strings: 要分词的字符串列表 / A list of strings to be tokenized.
            tokenizer: 可以编码和解码文本的 transformers.PreTrainedTokenizer 对象
                      A transformers.PreTrainedTokenizer object that can encode and decode text.

        Returns:
            包含以下键值对的字典 / A dictionary with the following keys and values:
                - input_ids: 包含分词后输入 ID 的 torch.LongTensor 对象列表，形状为 (max_length,)
                            A list of torch.LongTensor objects of shape (max_length,) containing the tokenized input ids.
                - labels: 包含分词后标签的 torch.LongTensor 对象列表，形状为 (max_length,)
                         A list of torch.LongTensor objects of shape (max_length,) containing the tokenized labels.
                - input_ids_lens: 表示填充前输入 ID 长度的整数列表
                                 A list of integers representing the lengths of the input ids before padding.
                - labels_lens: 表示填充前标签长度的整数列表
                              A list of integers representing the lengths of the labels before padding.
        """
        # 对每个字符串进行分词
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ) for text in strings
        ]
        # 提取输入 ID（这里 input_ids 和 labels 是相同的）
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        # 计算每个序列的实际长度（不包括填充标记）
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(self, sources, targets, tokenizer):
        """
        使用给定的分词器预处理源文本和目标文本。
        Preprocesses the sources and targets using the given tokenizer.

        Args:
            sources: 表示源文本的字符串列表 / A list of strings representing the source texts.
            targets: 表示目标文本的字符串列表 / A list of strings representing the target texts.
            tokenizer: 可以编码和解码文本的 transformers.PreTrainedTokenizer 对象
                      A transformers.PreTrainedTokenizer object that can encode and decode text.

        Returns:
            包含以下键值对的字典 / A dictionary with the following keys and values:
                - input_ids: 包含填充输入 ID 的 torch.LongTensor 对象列表，形状为 (max_length,)
                            A list of torch.LongTensor objects of shape (max_length,) containing the padded input ids.
                - labels: 包含填充标签的 torch.LongTensor 对象列表，形状为 (max_length,)
                         A list of torch.LongTensor objects of shape (max_length,) containing the padded labels.
        """
        # 将源文本和目标文本拼接
        examples = [s + t for s, t in zip(sources, targets)]
        # 分别对完整示例和源文本进行分词
        examples_tokenized, sources_tokenized = [
            self._tokenize_fn(strings, tokenizer)
            for strings in (examples, sources)
        ]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)  # 复制输入 ID 作为标签
        # 将源文本部分的标签设置为忽略索引（不参与损失计算）
        for label, source_len in zip(labels,
                                     sources_tokenized["input_ids_lens"]):
            label[:source_len] = DefaultToken.IGNORE_INDEX.value
        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        """返回数据集的大小 / Returns the size of the dataset"""
        return len(self.input_ids)

    def __getitem__(self, i):
        """获取数据集中的第 i 个样本 / Gets the i-th sample from the dataset"""
        # return dict(input_ids=self.input_ids[i],
        #             labels=self.labels[i],
        #             categories=self.categories[i])
        
        return dict(input_ids=self.input_ids[i],  # 输入 ID
                    labels=self.labels[i],  # 标签
                    categories=self.categories[i],  # 类别
                    instruction=self.list_data_dict[i]["instruction"], # 指令（为 GSM8K 等数据集添加）
                    input=self.list_data_dict[i]["input"], # 输入（为 GSM8K 等数据集添加）
                    output=self.list_data_dict[i]["output"], # 输出（为 GSM8K 等数据集添加）
                    )
