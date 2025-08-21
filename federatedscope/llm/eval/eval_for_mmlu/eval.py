# ref: https://github.com/hendrycks/test/blob/master/evaluate_flan.py
"""
MMLU (Massive Multitask Language Understanding) 评估模块

该模块实现了对大型语言模型在 MMLU 基准测试上的评估功能。
MMLU 是一个涵盖 57 个学科的多选题测试，用于评估模型的知识理解能力。

主要功能：
- 格式化 MMLU 测试题目和选项
- 生成包含少样本示例的提示文本
- 执行模型推理和答案提取
- 计算各学科和总体准确率
- 支持不同学科类别的分组统计

参考：https://github.com/hendrycks/test/blob/master/evaluate_flan.py
"""

import os
import torch
import numpy as np
import pandas as pd
from federatedscope.llm.eval.eval_for_mmlu.categories import \
     subcategories, categories
import json
import transformers

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.misc.fschat import FSChatBot
from federatedscope.core.data.utils import download_url
import tarfile

transformers.logging.set_verbosity(40)

# MMLU 多选题的选项标识
choices = ["A", "B", "C", "D"]


def format_subject(subject):
    """
    格式化学科名称，将下划线分隔的学科名转换为可读格式。
    
    Args:
        subject (str): 原始学科名称（如 "abstract_algebra"）
        
    Returns:
        str: 格式化后的学科名称（如 " abstract algebra"）
    """
    ll = subject.split("_")
    s = ""
    for entry in ll:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    """
    格式化单个 MMLU 题目为标准的多选题格式。
    
    Args:
        df (pd.DataFrame): 包含题目数据的 DataFrame
        idx (int): 题目在 DataFrame 中的索引
        include_answer (bool): 是否包含答案，默认为 True
        
    Returns:
        str: 格式化后的题目文本，包含问题、选项和答案（可选）
    """
    prompt = df.iloc[idx, 0]  # 题目问题
    k = df.shape[1] - 2  # 选项数量
    # 添加所有选项
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    # 如果需要包含答案
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    """
    生成包含少样本示例的 MMLU 提示文本。
    
    Args:
        train_df (pd.DataFrame): 训练样本数据
        subject (str): 学科名称
        k (int): 使用的示例数量，-1 表示使用所有样本
        
    Returns:
        str: 完整的提示文本，包含学科说明和示例题目
    """
    # 生成学科介绍
    prompt = "The following are multiple choice \
        questions (with answers) about {}.\n\n".format(format_subject(subject))
    # 确定示例数量
    if k == -1:
        k = train_df.shape[0]
    # 添加所有示例
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(subject, model, tokenizer, dev_df, test_df, device):
    """
    评估模型在特定学科上的 MMLU 性能。
    
    该函数对给定学科的所有测试题目进行推理，计算模型的准确率。
    使用少样本学习的方式，在每个测试题目前添加训练样本作为示例。
    
    Args:
        subject (str): 学科名称
        model: 待评估的语言模型
        tokenizer: 对应的分词器
        dev_df (pd.DataFrame): 开发集数据，用作少样本示例
        test_df (pd.DataFrame): 测试集数据
        device: 计算设备（CPU 或 GPU）
        
    Returns:
        tuple: (正确性列表, 准确率, 概率分布列表)
    """
    cors = []  # 存储每题的正确性
    all_probs = []  # 存储每题的概率分布
    answers = choices[:test_df.shape[1] - 2]  # 可用的答案选项

    for i in range(test_df.shape[0]):
        # 构建提示文本并确保长度适合
        k = 5  # 初始少样本数量
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        # 编码输入文本
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
        ).input_ids.to(device)

        # 如果输入过长，减少少样本数量
        while input_ids.shape[-1] > 1024:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt,
                                  return_tensors="pt").input_ids.to(device)

        # 获取正确答案
        label = test_df.iloc[i, test_df.shape[1] - 1]

        # 模型推理，获取最后一个位置的 logits
        logits = model(input_ids=input_ids).logits[0, -1]

        # 计算各选项的概率分布
        probs = (torch.nn.functional.softmax(
            torch.tensor([
                logits[tokenizer("A").input_ids[-1]],
                logits[tokenizer("B").input_ids[-1]],
                logits[tokenizer("C").input_ids[-1]],
                logits[tokenizer("D").input_ids[-1]],
            ]).float(),
            dim=0,
        ).detach().cpu().numpy())
        # 选择概率最高的答案
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        # 判断是否正确
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    # 计算准确率
    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main():
    init_cfg = global_cfg.clone()
    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to
    fschatbot = FSChatBot(init_cfg)
    tokenizer = fschatbot.tokenizer
    model = fschatbot.model
    device = fschatbot.device

    if not os.path.exists("data/mmlu"):
        download_url("https://people.eecs.berkeley.edu/~hendrycks/data.tar",
                     init_cfg.data.root)
        t = tarfile.open("data/data.tar", "r:")
        os.makedirs("data/mmlu/")
        t.extractall(path="data/mmlu/")
        t.close()

    data_dir = os.path.join(init_cfg.data.root, "mmlu/data")
    eval_dir = "eval_result"

    subjects = sorted([
        f.split("_test.csv")[0]
        for f in os.listdir(os.path.join(data_dir, "test")) if "_test.csv" in f
    ])

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    if not os.path.exists(
            os.path.join(eval_dir, "results_{}".format(
                init_cfg.federate.save_to))):
        os.makedirs(
            os.path.join(eval_dir,
                         "results_{}".format(init_cfg.federate.save_to)))

    all_cors = []
    subcat_cors = {
        subcat: []
        for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(data_dir, "dev",
                                          subject + "_dev.csv"),
                             header=None)[:5]
        test_df = pd.read_csv(os.path.join(data_dir, "test",
                                           subject + "_test.csv"),
                              header=None)

        cors, acc, probs = eval(subject, model, tokenizer, dev_df, test_df,
                                device)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(init_cfg.federate.save_to)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(init_cfg.federate.save_to,
                                               choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(eval_dir,
                         "results_{}".format(init_cfg.federate.save_to),
                         "{}.csv".format(subject)),
            index=None,
        )

    results = {"subcategories": {}, "categories": {}}
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))

    results_file = os.path.join(
        eval_dir, "accuracies_{}.json".format(
            init_cfg.federate.save_to.replace("/", "_")))
    with open(results_file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
