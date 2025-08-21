"""
代码搜索和理解评估模块

该模块实现了对大型语言模型在代码搜索和理解任务上的评估功能。
主要用于评估模型理解自然语言描述并匹配相应代码片段的能力。

主要功能：
- 支持 CodeSearchNet 数据集的评估
- 实现少样本学习的代码搜索任务
- 计算模型在代码-文档匹配任务上的准确率
- 提供预定义的示例样本用于少样本学习

评估任务：
给定自然语言描述，从候选代码片段中选择最匹配的代码。
"""

import os
import torch
import random
import transformers
import numpy as np
from tqdm import tqdm

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_json, load_jsonl
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)

# 评估数据集名称
EVAL_DATA = 'code_search_net'  # code_search_net
# 少样本学习的示例数量
N_SHOT = 5
# 预定义的少样本学习示例，包含文档描述、代码片段和标签
SAMPLES = [{
    "idx": "cosqa-train-0",
    "doc": "python code to write bool value 1",
    "code": "def writeBoolean(self, n):\n        \"\"\"\n"
    "        Writes a Boolean to the stream.\n        "
    "\"\"\"\n        t = TYPE_BOOL_TRUE\n\n        "
    "if n is False:\n            t = TYPE_BOOL_FALSE\n\n"
    "        self.stream.write(t)",
    "label": 0
}, {
    "idx": "cosqa-train-9",
    "doc": "1d array in char datatype in python",
    "code": "def _convert_to_array(array_like, dtype):\n"
    "        \"\"\"\n        "
    "Convert Matrix attributes which are "
    "array-like or buffer to array.\n        "
    "\"\"\"\n        if isinstance(array_like, bytes):\n"
    "            return np.frombuffer(array_like, dtype=dtype)\n"
    "        return np.asarray(array_like, dtype=dtype)",
    "label": 1
}, {
    "idx": "cosqa-train-2",
    "doc": "python colored output to html",
    "code": "def _format_json(data, theme):\n    "
    "\"\"\"Pretty print a dict as a JSON, "
    "with colors if pygments is present.\"\"\"\n    "
    "output = json.dumps(data, indent=2, sort_keys=True)\n\n"
    "    if pygments and sys.stdout.isatty():\n        "
    "style = get_style_by_name(theme)\n        "
    "formatter = Terminal256Formatter(style=style)\n        "
    "return pygments.highlight(output, JsonLexer(), formatter)\n\n"
    "    return output",
    "label": 0
}, {
    "idx": "cosqa-train-18",
    "doc": "python condition non none",
    "code": "def _not(condition=None, **kwargs):\n    \"\"\"\n"
    "    Return the opposite of input condition.\n\n    "
    ":param condition: condition to process.\n\n    "
    ":result: not condition.\n    :rtype: bool\n    "
    "\"\"\"\n\n    result = True\n\n    "
    "if condition is not None:\n        "
    "result = not run(condition, **kwargs)\n\n    "
    "return result",
    "label": 1
}, {
    "idx": "cosqa-train-4",
    "doc": "python column of an array",
    "code": "def _vector_or_scalar(x, type='row'):\n    "
    "\"\"\"Convert an object to either a scalar or "
    "a row or column vector.\"\"\"\n    "
    "if isinstance(x, (list, tuple)):\n        "
    "x = np.array(x)\n    if isinstance(x, np.ndarray):\n"
    "        assert x.ndim == 1\n        "
    "if type == 'column':\n            "
    "x = x[:, None]\n    return x",
    "label": 0
}]


def build_prompt(sample, n_shot):
    """
    构建代码搜索任务的提示文本。
    
    该函数创建包含任务说明、少样本示例和待评估样本的完整提示文本。
    任务是判断给定的代码片段和文档描述之间的匹配程度。
    
    Args:
        sample (dict): 待评估的样本，包含 'category'、'instruction' 和 'input' 字段
        n_shot (int): 使用的少样本示例数量
        
    Returns:
        str: 完整的提示文本，包含任务说明、示例和待评估样本
    """
    # 任务说明
    input_text_prompt = 'Input: a piece of code and a document\n' \
                        'Output: 0 or 1 score indicating the degree of ' \
                        'matching between the code and the document, ' \
                        'with 0 indicating a mismatch ' \
                        'and 1 indicating a match.\n\n'

    # 随机选择少样本示例
    index_list = list(range(len(SAMPLES)))
    random.shuffle(index_list)
    for i in index_list[:n_shot]:
        input_text_prompt += f"Document: {SAMPLES[i]['doc']}\n" \
                             f"Code: {SAMPLES[i]['code']}\n" \
                             f"Score: {SAMPLES[i]['label']}\n\n"
    
    # 添加待评估的样本
    input_text_prompt += f"Document:{sample['category']}" \
                         f" {sample['instruction']}\n" \
                         f"Code: {sample['input']}\n" \
                         f"Score: "

    return input_text_prompt


@torch.no_grad()
def main():
    """
    代码搜索任务的主评估函数。
    
    该函数执行完整的代码搜索评估流程，包括：
    1. 配置初始化和参数解析
    2. 模型加载
    3. 测试数据准备（支持 CoSQA 和 CodeSearchNet 数据集）
    4. 批量推理和评估
    5. 按编程语言分类统计结果
    """
    # 初始化配置
    init_cfg = global_cfg.clone()
    args = parse_args()

    # 合并配置文件和命令行参数
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    # 设置日志和随机种子
    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # 加载微调后的模型（保存为 xxx.ckpt 格式）
    # 模型路径在 yaml 文件的 federate.save_to 中指定
    fschatbot = FSChatBot(init_cfg)
    tokenizer = fschatbot.tokenizer
    model = fschatbot.model
    device = fschatbot.device

    # 获取测试文件
    if EVAL_DATA == 'cosqa':
        # 加载 CoSQA 数据集
        fp = os.path.join(init_cfg.data.root, 'cosqa-dev.json')
        if not os.path.exists(fp):
            download_url(
                'https://github.com/microsoft/CodeXGLUE/raw/'
                'd67dd5c73b9c433307d7df5f9faab2af9f5d1742/'
                'Text-Code/NL-code-search-WebQuery/CoSQA/cosqa-dev.json',
                init_cfg.data.root)
        list_data_dict = load_json(fp,
                                   instruction='doc',
                                   input='code',
                                   output='label')
        # 为 CoSQA 数据添加编程语言类别
        for sample in list_data_dict:
            sample['category'] = 'python'
    elif EVAL_DATA == 'code_search_net':
        # 加载 CodeSearchNet 数据集
        fp = os.path.join(init_cfg.data.root, 'csn_test.jsonl')
        if not os.path.exists(fp):
            raise FileNotFoundError('Run `python '
                                    'federatedscope/llm/'
                                    'dataset/code_search_net.py` '
                                    'to build test file')
        list_data_dict = load_jsonl(fp,
                                    instruction='instruction',
                                    input='input',
                                    output='output',
                                    category='category')
    else:
        raise ValueError(EVAL_DATA)

    # 开始评估过程
    labels, preds, cors = [], [], []
    category = None
    for sample in tqdm(list_data_dict):
        # 当遇到新的编程语言类别时，打印上一类别的结果
        if sample['category'] != category:
            print(f"==============={category}===============\n"
                  f"Num of total question: {len(cors)}\n"
                  f"Average accuracy {np.mean(cors)}\n\n")
            category = sample['category']
            labels, preds, cors = [], [], []

        # 构建提示文本
        n_shot = N_SHOT
        input_text = build_prompt(sample, n_shot)
        label = sample['output']

        # 如果输入过长，减少少样本数量
        while len(input_text) > 1024 and n_shot > 0:
            n_shot -= 1
            input_text = build_prompt(sample, n_shot)

        # 编码输入并进行推理
        input_ids = \
            tokenizer(input_text, return_tensors="pt",
                      max_length=tokenizer.model_max_length).input_ids.to(
                device)
        logits = model(input_ids=input_ids).logits[0, -1]
        
        # 计算 0 和 1 的概率分布
        probs = (torch.nn.functional.softmax(
            torch.tensor([
                logits[tokenizer("0").input_ids[-1]],
                logits[tokenizer("1").input_ids[-1]],
            ]).float(),
            dim=0,
        ).detach().cpu().numpy())

        # 选择概率最高的预测
        pred = {0: 0, 1: 1}[np.argmax(probs)]

        # 判断是否正确
        cor = pred == label

        labels.append(label)
        preds.append(pred)
        cors.append(cor)

    # 打印最后一个类别的结果
    print(f"==============={category}===============\n"
          f"Num of total question: {len(cors)}\n"
          f"Average accuracy {np.mean(cors)}\n\n")


if __name__ == "__main__":
    main()
