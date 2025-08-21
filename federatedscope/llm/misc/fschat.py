"""
FederatedScope 聊天机器人模块

该模块实现了基于大型语言模型的聊天机器人功能，支持联邦学习训练的模型。
主要用于与用户进行自然语言对话，支持历史对话记录和提示模板。

主要功能：
- 加载预训练或微调后的语言模型
- 支持对话历史管理
- 提供多种文本生成方法（predict 和 generate）
- 支持自定义提示模板
- 支持特殊标记处理

类：
- FSChatBot: 主要的聊天机器人类，封装了模型加载、对话管理和文本生成功能
"""

import sys
import logging
import torch
import transformers
import copy

transformers.logging.set_verbosity(40)

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.llm.dataloader.dataloader import get_tokenizer
from federatedscope.llm.model.model_builder import get_llm
from federatedscope.llm.dataset.llm_dataset import PROMPT_DICT
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger

logger = logging.getLogger(__name__)


class FSChatBot(object):
    """
    基于语言模型的聊天机器人类。

    该类实现了一个可以与用户进行自然语言交互的聊天机器人。
    它使用预训练的语言模型作为基础，可以选择性地加载联邦学习微调后的检查点。
    支持使用历史对话和提示模板来提升对话质量。
    提供两种文本生成方法：predict 和 generate。

    属性:
        tokenizer: transformers.PreTrainedTokenizer 对象，用于文本编码和解码
        model: transformers.PreTrainedModel 对象，用于文本生成
        device: 字符串，表示模型运行的设备
        add_special_tokens: 布尔值，是否在输入输出文本中添加特殊标记
        max_history_len: 整数，表示用作上下文的最大历史轮次数
        max_len: 整数，表示每次响应生成的最大标记数
        history: 列表的列表，包含之前轮次的标记化输入和输出文本
    """
    def __init__(self, config):
        """
        使用给定配置初始化聊天机器人。

        参数:
            config: FederatedScope 配置对象，包含聊天机器人的各种设置
        """
        # 解析模型名称和模型中心
        model_name, model_hub = config.model.type.split('@')
        # 获取分词器
        self.tokenizer, _ = get_tokenizer(model_name, config.data.root,
                                          config.llm.tok_len, model_hub)
        # 构建语言模型
        self.model = get_llm(config)

        # 设置设备和特殊标记处理
        self.device = f'cuda:{config.device}'
        self.add_special_tokens = True

        # 加载模型权重（联邦学习微调或离线调优）
        if config.llm.offsite_tuning.use:
            # 使用离线调优模式
            from federatedscope.llm.offsite_tuning.utils import \
                wrap_offsite_tuning_for_eval
            self.model = wrap_offsite_tuning_for_eval(self.model, config)
        else:
            # 尝试加载联邦学习训练的检查点
            try:
                ckpt = torch.load(config.federate.save_to, map_location='cpu')
                if 'model' and 'cur_round' in ckpt:
                    self.model.load_state_dict(ckpt['model'])
                else:
                    self.model.load_state_dict(ckpt)
            except Exception as error:
                print(f"{error}, will use raw model.")

        # 启用半精度推理（如果配置）
        if config.train.is_enable_half:
            self.model.half()

        # 将模型移动到指定设备并设置为评估模式
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        # 使用 torch.compile 优化（PyTorch 2.0+）
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        # 设置对话参数
        self.max_history_len = config.llm.chat.max_history_len
        self.max_len = config.llm.chat.max_len
        self.history = []

    def _build_prompt(self, input_text):
        """
        为输入文本构建提示模板。

        参数:
            input_text: 字符串，表示用户的输入文本

        返回:
            字符串，表示使用提示模板格式化后的源文本
        """
        source = {'instruction': input_text}
        return PROMPT_DICT['prompt_no_input'].format_map(source)

    def predict(self, input_text, use_history=True, use_prompt=True):
        """
        使用模型为输入文本生成响应。

        参数:
            input_text: 字符串，表示用户的输入文本
            use_history: 布尔值，是否使用之前的对话轮次作为生成响应的上下文，默认为 True
            use_prompt: 布尔值，是否使用提示模板来创建源文本，默认为 True

        返回:
            字符串，表示聊天机器人的响应文本
        """
        # 如果使用提示模板，则格式化输入文本
        if use_prompt:
            input_text = self._build_prompt(input_text)
        # 编码输入文本并添加到历史记录
        text_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        self.history.append(text_ids)
        
        # 构建输入序列（包含历史上下文或仅当前输入）
        input_ids = []
        if use_history:
            # 使用最近的历史对话作为上下文
            for history_ctx in self.history[-self.max_history_len:]:
                input_ids.extend(history_ctx)
        else:
            # 仅使用当前输入
            input_ids.extend(text_ids)
        
        # 转换为张量并移动到设备
        input_ids = torch.tensor(input_ids).long()
        input_ids = input_ids.unsqueeze(0).to(self.device)
        
        # 生成响应
        response = self.model.generate(input_ids=input_ids,
                                       max_new_tokens=self.max_len,
                                       num_beams=4,
                                       no_repeat_ngram_size=2,
                                       early_stopping=True,
                                       temperature=0.0)

        # 将响应添加到历史记录并解码
        self.history.append(response[0].tolist())
        response_tokens = \
            self.tokenizer.decode(response[0][input_ids.shape[1]:],
                                  skip_special_tokens=True)
        return response_tokens

    @torch.no_grad()
    def generate(self, input_text, generate_kwargs={}):
        """
        使用模型和额外参数为输入文本生成响应。

        参数:
            input_text: 字符串，表示用户的输入文本
            generate_kwargs: 字典，传递给模型 generate 方法的关键字参数，默认为空字典

        返回:
            字符串或字符串列表，表示聊天机器人的响应文本。
            如果 generate_kwargs 包含 num_return_sequences > 1，则返回字符串列表，
            否则返回单个字符串。
        """
        # 编码输入文本
        input_text = self.tokenizer(
            input_text,
            padding=False,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = input_text.input_ids.to(self.device)
        attention_mask = input_text.attention_mask.to(self.device)

        # 使用自定义参数生成响应
        output_ids = self.model.generate(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         **generate_kwargs)
        
        # 解码所有生成的响应
        response = []
        for i in range(output_ids.shape[0]):
            response.append(
                self.tokenizer.decode(output_ids[i][input_ids.shape[1]:],
                                      skip_special_tokens=True,
                                      ignore_tokenization_space=True))

        # 返回单个响应或响应列表
        if len(response) > 1:
            return response
        return response[0]

    def clear(self):
        """清除之前对话轮次的历史记录。

        该方法可用于重置聊天机器人的状态并开始新的对话。
        """
        self.history = []


class FSChatBot_My(FSChatBot):
    """
    自定义聊天机器人类，继承自 FSChatBot。
    
    该类允许直接传入已有的模型实例，而不是从配置中重新构建模型。
    主要用于在已有模型基础上快速创建聊天机器人实例。
    """
    def __init__(self, model, config):
        """
        使用给定的模型和配置初始化自定义聊天机器人。
        
        参数:
            model: 预训练的语言模型实例
            config: FederatedScope 配置对象
        """
        # 解析模型名称和模型中心
        model_name, model_hub = config.model.type.split('@')
        # 获取分词器
        self.tokenizer, _ = get_tokenizer(model_name, config.data.root,
                                          config.llm.tok_len, model_hub)
        # 直接复制传入的模型，而不是从检查点加载
        self.device = f'cuda:{config.eval_device}'
        self.model = copy.deepcopy(model).to(self.device)   # 直接复制，不从检查点加载
        
        self.add_special_tokens = True

        # 如果使用离线调优，则包装模型
        if config.llm.offsite_tuning.use:
            from federatedscope.llm.offsite_tuning.utils import \
                wrap_offsite_tuning_for_eval
            self.model = wrap_offsite_tuning_for_eval(self.model, config)

        # 启用半精度推理（如果配置）
        if config.train.is_enable_half:
            self.model.half()

        # 设置为评估模式并优化
        self.model = self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        # 设置对话参数
        self.max_history_len = config.llm.chat.max_history_len
        self.max_len = config.llm.chat.max_len
        self.history = []


def main():
    """
    主函数，启动 FederatedScope 聊天机器人的交互式会话。
    
    该函数负责：
    1. 解析命令行参数和配置文件
    2. 初始化日志和随机种子
    3. 创建聊天机器人实例
    4. 启动交互式对话循环
    """
    # 克隆全局配置
    init_cfg = global_cfg.clone()
    # 解析命令行参数
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    # 更新日志配置和设置随机种子
    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # 创建聊天机器人实例
    chat_bot = FSChatBot(init_cfg)
    # 显示欢迎信息
    welcome = "Welcome to FSChatBot，" \
              "`clear` to clear history，" \
              "`quit` to end chat."
    print(welcome)
    while True:
        input_text = input("\nUser:")
        if input_text.strip() == "quit":
            break
        if input_text.strip() == "clear":
            chat_bot.clear()
            print(welcome)
            continue
        print(f'\nFSBot: {chat_bot.predict(input_text)}')


if __name__ == "__main__":
    main()
