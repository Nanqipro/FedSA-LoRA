# -*- coding: utf-8 -*-
"""
GSM8K 数学推理任务评估模块

该模块实现了对 GSM8K（Grade School Math 8K）数据集的评估功能，用于测试大语言模型
在小学数学应用题上的推理能力。GSM8K 包含 8500 个高质量的小学数学应用题，
每个问题都需要 2-8 步的推理过程来解决。

主要功能：
1. 提取模型输出中的数值答案
2. 验证模型答案的正确性
3. 构建少样本学习的演示文本
4. 构建包含思维链（Chain-of-Thought）的提示
5. 清理和标准化模型预测结果
6. 执行完整的 GSM8K 评估流程

参考：https://github.com/kojima-takeshi188/zero_shot_cot
"""

import re
import os
import random
import transformers
from tqdm import tqdm

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)

# 用于提取答案的正则表达式，匹配 "#### 数字" 格式
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
# 无效答案的标识符
INVALID_ANS = "[invalid]"

# 少样本学习的示例数量
N_SHOT = 8
# 是否启用思维链（Chain-of-Thought）推理
COT_FLAG = True
# 调试模式标志
DEBUG = False
# 答案触发词，用于标识最终答案
ANSWER_TRIGGER = "The answer is"


def extract_answer_from_output(completion):
    """
    从模型输出中提取数值答案。
    
    该函数使用正则表达式从模型的完整输出中提取最终的数值答案。
    GSM8K 数据集中的答案格式为 "#### 数字"，该函数专门用于解析这种格式。
    
    Args:
        completion (str): 模型的完整输出文本
        
    Returns:
        str: 提取到的数值答案字符串，如果未找到则返回 "[invalid]"
    """
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")  # 移除数字中的逗号
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    """
    判断模型答案是否正确。
    
    该函数将模型预测的答案与标准答案进行比较，判断是否正确。
    会自动处理答案格式的转换，将 "The answer is" 格式转换为 "####" 格式。
    
    Args:
        model_answer (str): 模型预测的答案
        answer (str): 标准答案（可能包含 "The answer is" 或 "####" 格式）
        
    Returns:
        bool: 如果模型答案正确返回 True，否则返回 False
    """
    # 将答案转换回原始格式
    if 'The answer is' in answer:
        answer = answer.replace('The answer is', '####')
    
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def create_demo_text(n_shot=8, cot_flag=True):
    """
    创建少样本学习的演示文本。
    
    该函数构建用于少样本学习的演示样例，包含预定义的数学问题、推理过程和答案。
    支持思维链（Chain-of-Thought）推理模式，可以展示完整的推理步骤。
    
    Args:
        n_shot (int): 使用的演示样例数量，默认为 8
        cot_flag (bool): 是否启用思维链推理，默认为 True
        
    Returns:
        str: 格式化的演示文本，包含问题、推理过程（如果启用）和答案
    """
    question, chain, answer = [], [], []
    question.append("There are 15 trees in the grove. "
                    "Grove workers will plant trees in the grove today. "
                    "After they are done, there will be 21 trees. "
                    "How many trees did the grove workers plant today?")
    chain.append("There are 15 trees originally. "
                 "Then there were 21 trees after some more were planted. "
                 "So there must have been 21 - 15 = 6.")
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?")
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?")
    chain.append("Originally, Leah had 32 chocolates. "
                 "Her sister had 42. So in total they had 32 + 42 = 74. "
                 "After eating 35, they had 74 - 35 = 39.")
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?")
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8.")
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?")
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9.")
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?")
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29.")
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?")
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls.")
    answer.append("33")

    question.append("Olivia has $23. She bought five bagels for $3 each. "
                    "How much money does she have left?")
    chain.append("Olivia had 23 dollars. "
                 "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
                 "So she has 23 - 15 dollars left. 23 - 15 is 8.")
    answer.append("8")

    # 随机化示例的顺序
    index_list = list(range(len(question)))
    random.shuffle(index_list)

    # 拼接演示样例
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            # 思维链模式：包含完整的推理过程
            demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
        else:
            # 直接回答模式：只包含问题和答案
            demo_text += "Question: " + question[i] + "\nAnswer: " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    return demo_text


def build_prompt(input_text, n_shot, cot_flag):
    """
    构建完整的提示文本，包含演示样例和待解决的问题。
    
    该函数将演示样例和用户输入的问题组合成完整的提示文本，
    用于指导模型进行数学推理。
    
    Args:
        input_text (str): 待解决的数学问题
        n_shot (int): 使用的演示样例数量
        cot_flag (bool): 是否启用思维链推理
        
    Returns:
        str: 完整的提示文本，包含演示样例和问题
    """
    demo = create_demo_text(n_shot, cot_flag)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt


def clean_answer(model_pred):
    """
    清理和标准化模型预测的答案。
    
    该函数从模型的原始输出中提取并清理数值答案，处理各种格式问题，
    如逗号分隔符、句号结尾等，确保答案格式的一致性。
    
    Args:
        model_pred (str): 模型的原始预测输出
        
    Returns:
        str: 清理后的数值答案，如果无法提取则返回 "[invalid]"
    """
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # 如果有答案触发词，选择第一个答案
        pred = preds[1]
    else:
        # 如果没有答案触发词，选择最后一个数字
        pred = preds[-1]

    pred = pred.replace(",", "")  # 移除逗号
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]  # 提取所有数字

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # 有答案标志时选择第一个数字
        pred = pred[0]
    else:
        # 无答案标志时选择最后一个数字
        pred = pred[-1]

    # 对于算术任务，如果数字以句号结尾，则省略句号
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred


def main():
    """
    GSM8K 数学推理任务的主评估函数。
    
    该函数执行完整的 GSM8K 评估流程，包括：
    1. 配置初始化和参数解析
    2. 模型加载
    3. 测试数据准备
    4. 批量推理和评估
    5. 结果统计和输出
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

    # 获取测试文件
    fp = os.path.join(init_cfg.data.root, 'gsm8k_test.jsonl')
    if not os.path.exists(fp):
        # 下载 GSM8K 测试数据
        download_url(
            'https://raw.githubusercontent.com/openai/'
            'grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/'
            'grade_school_math/data/test.jsonl', init_cfg.data.root)
        os.rename(os.path.join(init_cfg.data.root, 'test.jsonl'), fp)

    # 加载测试数据
    list_data_dict = load_jsonl(fp, instruction='question', output='answer')

    # 开始评估过程
    answers = []
    for sample in tqdm(list_data_dict):
        # 构建包含少样本示例的提示文本
        input_text = build_prompt(sample['instruction'], N_SHOT, COT_FLAG)
        # 设置生成参数
        generate_kwargs = dict(max_new_tokens=256, top_p=0.95, temperature=0.8)
        # 生成模型回答
        model_completion = fschatbot.generate(input_text, generate_kwargs)
        # 清理和提取答案
        model_answer = clean_answer(model_completion)
        # 判断答案是否正确
        is_cor = is_correct(model_answer, sample['output'])
        answers.append(is_cor)
        
        # 调试模式下打印完整输入
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')
        
        # 打印评估详情
        print(f'Question: {sample["instruction"]}\n\n'
              f'Answers: {extract_answer_from_output(sample["output"])}\n\n'
              f'Model Answers: {model_answer}\n\n'
              f'Model Completion: {model_completion}\n\n'
              f'Is correct: {is_cor}\n\n')

        # 打印当前统计结果
        print(f'Num of total question: {len(answers)}, '
              f'correct num: {sum(answers)}, '
              f'correct rate: {float(sum(answers))/len(answers)}.')


if __name__ == "__main__":
    main()
