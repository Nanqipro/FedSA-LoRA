# 导入必要的库
# -*- coding: utf-8 -*-
"""
LLM 数据加载器模块

该模块提供了用于大语言模型（LLM）训练的数据加载和处理功能，包括：
1. LLMDataCollator：用于批量数据整理的数据收集器
2. get_tokenizer：分词器加载和配置函数
3. load_json/load_jsonl：JSON 和 JSONL 格式数据加载函数
4. load_llm_dataset：主要的数据集加载函数，支持多种数据集类型

支持的数据集类型包括：
- JSON/JSONL 格式的自定义数据集
- Alpaca 和 Alpaca Cleaned 指令数据集
- Dolly-15K 对话数据集
- GSM8K 数学推理数据集
- CodeSearchNet 代码搜索数据集
- Rosetta Alpaca 代码生成数据集
"""

import os
import gzip
import json
import random
import logging
import torch
import transformers

from dataclasses import dataclass
from federatedscope.llm.dataset.llm_dataset import DefaultToken, LLMDataset
from federatedscope.core.data.utils import download_url


logger = logging.getLogger(__name__)


@dataclass
class LLMDataCollator(object):
    """
    用于语言模型监督微调的数据整理器。
    该类实现了一个可调用对象，接收实例列表并返回包含 input_ids、labels 和 attention_mask 张量的批次。
    input_ids 和 labels 分别用分词器的 pad_token_id 和特殊忽略索引值进行填充。
    attention_mask 指示哪些标记不是填充标记。
    
    A data collator for supervised fine-tuning of language models.
    This class implements a callable that takes a list of instances and
    returns a batch of input_ids, labels, and attention_mask tensors. The
    input_ids and labels are padded with the tokenizer's pad_token_id and a
    special ignore index value, respectively. The attention_mask indicates
    which tokens are not padding.
    """

    tokenizer: transformers.PreTrainedTokenizer  # 预训练的分词器

    def __call__(self, instances):
        """将实例列表整理成批次。
        
        Args:
            instances: 字典列表，每个字典包含 input_ids 和 labels 作为 torch.LongTensor 对象。
        
        Returns:
            包含以下键值对的字典：
                - input_ids: 形状为 (batch_size, max_length) 的 torch.LongTensor，
                    包含填充后的输入 ID。
                - labels: 形状为 (batch_size, max_length) 的 torch.LongTensor，
                    包含填充后的标签。
                - attention_mask: 形状为 (batch_size, max_length) 的 torch.BoolTensor，
                    指示哪些标记不是填充标记。
        
        Collates a list of instances into a batch.

        Args:
            instances: A list of dictionaries, each containing input_ids and
                labels as torch.LongTensor objects.

        Returns:
            A dictionary with the following keys and values:
                - input_ids: A torch.LongTensor of shape (batch_size,
                max_length)
                    containing the padded input ids.
                - labels: A torch.LongTensor of shape (batch_size, max_length)
                    containing the padded labels.
                - attention_mask: A torch.BoolTensor of shape (batch_size,
                max_length)
                    indicating which tokens are not padding.
        """
        # 从实例中提取 input_ids 和 labels
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # 使用分词器的 pad_token_id 填充 input_ids 到相同长度
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        # 使用忽略索引填充 labels 到相同长度
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=DefaultToken.IGNORE_INDEX.value)
        # return dict(
        #     input_ids=input_ids,
        #     labels=labels,
        #     attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        # )
        
        # 为评估 GSM8K 数据集添加的字段
        instruction = [instance['instruction'] for instance in instances]
        input = [instance['input'] for instance in instances]
        output = [instance['output'] for instance in instances]
        
        # 返回包含所有必要字段的批次字典
        return dict(
            input_ids=input_ids,  # 填充后的输入 ID
            labels=labels,  # 填充后的标签
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),  # 注意力掩码
            instruction=instruction,  # 指令文本（用于 GSM8K 评估）
            input=input,  # 输入文本（用于 GSM8K 评估）
            output=output,  # 输出文本（用于 GSM8K 评估）
        )


def get_tokenizer(model_name, cache_dir, tok_len=128, pkg='huggingface_llm'):
    """
    从预训练模型名称加载分词器，并添加默认的特殊标记（如果尚未定义）。
    还设置模型的最大长度和分词器的填充方向。
    
    Args:
        model_name: 字符串，预训练模型的名称。
        cache_dir: 字符串，缓存目录的路径。
        tok_len: 整数，分词器的最大长度，默认为 128。
        pkg: 字符串，使用的包类型，默认为 'huggingface_llm'。
    
    This function loads a tokenizer from a pretrained model name and adds some
    default special tokens if they are not already defined. It also sets the
    model max length and the padding side of the tokenizer.

    Args:
        model_name: A string, the name of the pretrained model.
        cache_dir: A string, the path to the cache directory.
        tok_len: An integer, the maximum length of the tokens. Defaults to 128.
    
    Returns:
        返回元组 (tokenizer, num_new_tokens)，其中：
            - tokenizer: transformers.AutoTokenizer 对象。
            - num_new_tokens: 整数，新添加的特殊标记数量。

    Returns:
        A tuple of (tokenizer, num_new_tokens), where:
            - tokenizer: A transformers.AutoTokenizer object.
            - num_new_tokens: An integer, the number of new special tokens
    """
    # 验证包类型是否支持
    assert pkg in ['huggingface_llm', 'modelscope_llm'], \
        f'Not supported package {pkg}.'

    # 根据包类型导入相应的 AutoTokenizer
    if pkg == 'huggingface_llm':
        from transformers import AutoTokenizer
    elif pkg == 'modelscope_llm':
        from modelscope import AutoTokenizer

    # 从预训练模型加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        model_max_length=tok_len,
        padding_side="right",
        use_fast=False,
    )

    # 检查并添加缺失的特殊标记
    special_tokens = dict()
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value

    # 添加特殊标记到分词器
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)

    return tokenizer, num_new_tokens


def load_json(file_path,
              instruction='instruction',
              input='input',
              output='output',
              category='category'):
    """
    读取包含示例列表的 JSON 文件，每个示例包含指令、输入、输出和类别。
    返回具有相同键的字典列表，但可以选择重命名它们。
    
    Args:
        file_path: 字符串，JSON 文件的路径。
        instruction: 字符串，指令字段的键名，默认为 'instruction'。
        input: 字符串，输入字段的键名，默认为 'input'。
        output: 字符串，输出字段的键名，默认为 'output'。
        category: 字符串，类别字段的键名，默认为 'category'。
    
    Returns:
        字典列表，每个字典包含四个键：instruction、input、output 和 category。
        值取自 JSON 文件，如果文件中不存在相应的键，则可能为 None。
    
    This function reads a JSON file that contains a list of examples,
    each with an instruction, an input, an output, and a category. It
    returns a list of dictionaries with the same keys, but with the
    option to rename them.

    Args:
        file_path: A string, the path to the JSON file.
        instruction: A string, the key for the instruction field. Defaults
            to 'instruction'.
        input: A string, the key for the input field. Defaults to 'input'.
        output: A string, the key for the output field. Defaults to 'output'.
        category: A string, the key for the category field. Defaults to
            'category'.

    Returns:
        A list of dictionaries, each with four keys: instruction, input,
            output, and category. The values are taken from the JSON file
            and may be None if the corresponding key is not present in the
            file.
    """

    # 格式：[{'instruction': ..., 'input': ..., 'output':...}]
    # 读取 JSON 文件
    with open(file_path, 'r', encoding="utf-8") as f:
        list_data_dict = json.load(f)

    # 替换键名并创建新的数据字典列表
    new_list_data_dict = []
    for item in list_data_dict:
        new_item = dict(
            instruction=item[instruction] if instruction in item else None,
            input=item[input] if input in item else None,
            output=item[output] if output in item else None,
            category=item[category] if category in item else None)
        new_list_data_dict.append(new_item)
    return new_list_data_dict


def load_jsonl(file_path,
               instruction='instruction',
               input='input',
               output='output',
               category='category',
               is_gzip=False):
    """
    读取 JSONL 文件，每行包含一个示例，每个示例包含指令、输入、输出和类别。
    返回具有相同键的字典列表，但可以选择重命名它们。还支持读取 gzip 压缩文件。
    
    Args:
        file_path: 字符串，JSONL 文件的路径。
        instruction: 字符串，指令字段的键名，默认为 'instruction'。
        input: 字符串，输入字段的键名，默认为 'input'。
        output: 字符串，输出字段的键名，默认为 'output'。
        category: 字符串，类别字段的键名，默认为 'category'。
        is_gzip: 布尔值，文件是否为 gzip 压缩格式，默认为 False。
    
    Returns:
        字典列表，每个字典包含四个键：instruction、input、output 和 category。
        值取自 JSONL 文件，如果行中不存在相应的键，则可能为 None。
    
    This function reads a JSONL file that contains one example per line,
    each with an instruction, an input, an output, and a category. It
    returns a list of dictionaries with the same keys, but with the option
    to rename them. It also supports reading gzip-compressed files.

    Args:
        file_path: A string, the path to the JSONL file.
        instruction: A string, the key for the instruction field. Defaults
            to 'instruction'.
        input: A string, the key for the input field. Defaults to 'input'.
        output: A string, the key for the output field. Defaults to 'output'.
        category: A string, the key for the category field. Defaults to
            'category'.
        is_gzip: A boolean, whether the file is gzip-compressed or not.
            Defaults to False.

    Returns:
        A list of dictionaries, each with four keys: instruction, input,
        output, and category. The values are taken from the JSONL file and
        may be None if the corresponding key is not present in the line.

    """
    # 每行的格式：{'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    # 根据是否为 gzip 压缩文件选择打开函数
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        for line in f:
            # 解析每行的 JSON 数据
            item = json.loads(line)
            # 创建新的字典项，重命名键并处理缺失值
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                input=item[input] if input in item else None,
                output=item[output] if output in item else None,
                category=item[category] if category in item else None)
            item = new_item
            list_data_dict.append(item)
    return list_data_dict


def load_llm_dataset(config=None, **kwargs):
    """
    加载 LLM 数据集的主函数，接受配置对象和可选关键字参数，返回数据集对象和更新的配置对象。
    该函数支持多种数据集类型，如 JSON、JSONL、alpaca、alpaca_cleaned、dolly-15K、
    gsm8k、code_search_net、rosetta_alpaca。如果数据目录中找不到数据文件，
    它将从相应的 URL 下载数据文件。还会从预训练模型名称加载分词器，
    并添加一些默认的特殊标记（如果尚未定义）。
    
    Args:
        config: 对象，加载数据集的配置。
        **kwargs: 可选关键字参数，可以覆盖配置属性。
    
    Returns:
        元组 (dataset, config)，其中：
            - dataset: LLMDataset 对象，包含具有 instruction、input、output 和 category 字段的示例。
            - config: 对象，更新后的配置。
    
    This function takes a config object and optional keyword arguments and
    returns a dataset object and an updated config object.
    The function supports various dataset types, such as JSON, JSONL, alpaca,
    alpaca_cleaned, dolly-15K, gsm8k, code_search_net, rosetta_alpaca. It
    will download the data files from their respective URLs if they are not
    found in the data directory. It will also load a tokenizer from a
    pretrained model name and add some default special tokens if they are
    not already defined.

    Args:
        config: An object, the configuration for loading the dataset.
        **kwargs: Optional keyword arguments that can override the config
            attributes.

    Returns:
        A tuple of (dataset, config), where:
            - dataset: A LLMDataset object that contains the examples with
                instruction, input, output, and category fields.
            - config: An object, the updated configuration.
    """
    # 解析模型名称和模型中心
    model_name, model_hub = config.model.type.split('@')
    # 获取分词器和新增标记数量
    tokenizer, num_new_tokens = \
        get_tokenizer(model_name, config.data.root, config.llm.tok_len,
                      model_hub)

    # 解析数据集名称
    dataset_name, _ = config.data.type.split('@')

    # 根据数据集类型加载相应的数据
    if dataset_name.endswith('.json'):
        # 加载 JSON 格式数据集
        fp = os.path.join(config.data.root, dataset_name)
        list_data_dict = load_json(fp)
        dataset = LLMDataset(list_data_dict, tokenizer)
    elif dataset_name.endswith('.jsonl'):
        # 加载 JSONL 格式数据集
        fp = os.path.join(config.data.root, dataset_name)
        list_data_dict = load_jsonl(fp)
        dataset = LLMDataset(list_data_dict, tokenizer)
    elif dataset_name.lower() == 'alpaca':
        # 加载 Alpaca 数据集
        fp = os.path.join(config.data.root, 'alpaca_data.json')
        download_url(
            'https://raw.githubusercontent.com/tatsu-lab'
            '/stanford_alpaca/'
            '761dc5bfbdeeffa89b8bff5d038781a4055f796a/'
            'alpaca_data.json', config.data.root)
        list_data_dict = load_json(fp)
        dataset = LLMDataset(list_data_dict, tokenizer)
    elif dataset_name.lower() == 'alpaca_cleaned':
        # 加载清理后的 Alpaca 数据集
        fp = os.path.join(config.data.root, 'alpaca_data_cleaned.json')
        download_url(
            'https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/'
            'a7d629079a95c2e4b7ec7dfe55087fbd18d9eba8/'
            'alpaca_data_cleaned.json', config.data.root)
        list_data_dict = load_json(fp)
        dataset = LLMDataset(list_data_dict, tokenizer)
    elif dataset_name.lower() == 'dolly-15k':
        # 加载 Dolly-15K 数据集
        fp = os.path.join(config.data.root, 'databricks-dolly-15k.jsonl')
        download_url(
            'https://raw.githubusercontent.com/databrickslabs'
            '/dolly/d000e3030970379aabbf6d291f50ffdd3b715b64'
            '/data/databricks-dolly-15k.jsonl', config.data.root)
        list_data_dict = load_jsonl(fp,
                                    instruction='instruction',
                                    input='context',
                                    output='response',
                                    category='category')
        dataset = LLMDataset(list_data_dict, tokenizer)
    elif dataset_name.lower() == 'gsm8k':
        # 加载 GSM8K 数学推理数据集
        fp = os.path.join(config.data.root, 'gsm8k_train.jsonl')
        if not os.path.exists(fp):
            # 下载训练数据
            download_url(
                'https://raw.githubusercontent.com/openai/grade-school-math'
                '/3101c7d5072418e28b9008a6636bde82a006892c/'
                'grade_school_math/data/train.jsonl', config.data.root)
            os.rename(os.path.join(config.data.root, 'train.jsonl'), fp)
        list_data_dict = load_jsonl(fp,
                                    instruction='question',
                                    output='answer')
        # 替换答案格式标记
        for i in range(len(list_data_dict)):
            list_data_dict[i]['output'] = \
                list_data_dict[i]['output'].replace('####', 'The answer is')
        # 创建训练数据集
        train_dataset = LLMDataset(list_data_dict, tokenizer)
        
        # 添加测试数据集
        fp_test = os.path.join(config.data.root, 'gsm8k_test.jsonl')
        if not os.path.exists(fp_test):
            # 下载测试数据
            download_url(
                'https://raw.githubusercontent.com/openai/'
                'grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/'
                'grade_school_math/data/test.jsonl', config.data.root)
            os.rename(os.path.join(config.data.root, 'test.jsonl'), fp)
        list_data_dict_test = load_jsonl(fp_test, 
                                         instruction='question', 
                                         output='answer')
        # 测试数据集不需要替换答案格式
        # for i in range(len(list_data_dict_test)):
        #     list_data_dict_test[i]['output'] = \
        #         list_data_dict_test[i]['output'].replace('####', 'The answer is')
        test_dataset = LLMDataset(list_data_dict_test, tokenizer)

        # 返回训练集、测试集和验证集（空）
        dataset = (train_dataset, test_dataset, [])
        
    elif dataset_name.lower() == 'code_search_net':
        # 加载 CodeSearchNet 代码搜索数据集
        from tqdm import tqdm
        from federatedscope.llm.dataset.code_search_net import \
            CSN_FILE_NUM_DICT

        list_data_dict = []
        logger.info('Loading code search net data file...')
        try:
            # 遍历所有编程语言
            for language in tqdm(CSN_FILE_NUM_DICT.keys()):
                sub_list_data_dict = []
                # 加载每种语言的所有训练文件
                for file_index in range(CSN_FILE_NUM_DICT[language]['train']):
                    fp = \
                        os.path.join(config.data.root, language,
                                     'final', 'jsonl', 'train',
                                     f'{language}_train_{file_index}.jsonl.gz')
                    # 加载 gzip 压缩的 JSONL 文件
                    tmp_list_data_dict = load_jsonl(
                        fp,
                        instruction='docstring',
                        input='language',
                        output='code',
                        category='language',
                        is_gzip=True,
                    )
                    sub_list_data_dict += tmp_list_data_dict
                # 对数据进行子采样
                raw_size = len(sub_list_data_dict)
                num_subsample = int(raw_size * config.data.subsample)
                list_data_dict += random.sample(sub_list_data_dict,
                                                num_subsample)
                logger.info(f"Subsample "
                            f"{sub_list_data_dict[0]['category']} with "
                            f"rate {config.data.subsample}: "
                            f"the sample size is # {num_subsample} "
                            f"(the raw size is {raw_size}).")
            # 在指令前添加编程语言信息
            for sample in list_data_dict:
                sample['instruction'] = \
                    sample['category'] + ' ' + sample['instruction']
        except FileNotFoundError:
            # 数据文件未找到时的错误处理
            raise FileNotFoundError(
                'Data not found! Please run `python '
                'federatedscope/llm/dataset/code_search_net.py` '
                'to download data.')
        dataset = LLMDataset(list_data_dict, tokenizer)
    elif dataset_name.lower() == 'rosetta_alpaca':
        # 加载 Rosetta Alpaca 代码数据集
        fp = os.path.join(config.data.root, 'rosetta_alpaca.json')
        download_url(
            'https://raw.githubusercontent.com/'
            'sahil280114/codealpaca/'
            'd269da106a579a623a654529b3cb91b5dfa9c72f/'
            'data/rosetta_alpaca.json', config.data.root)
        list_data_dict = load_json(fp,
                                   instruction='instruction',
                                   input='input',
                                   output='output',
                                   category='input')
        # 如果使用 meta 分割器，移除样本数量过少的 X86-64 Assembly
        if config.data.splitter == 'meta':
            list_data_dict = [
                i for i in list_data_dict if i['category'] != 'X86-64 Assembly'
            ]
        dataset = LLMDataset(list_data_dict, tokenizer)
    else:
        # 不支持的数据集类型
        raise ValueError(f'Not support data type {dataset_name}.')

    # 返回数据集和更新后的配置
    return dataset, config
