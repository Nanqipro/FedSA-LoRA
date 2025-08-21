"""
FederatedScope 大型语言模型训练器模块

该模块实现了专门用于大型语言模型训练的训练器类，支持联邦学习场景下的 LLM 训练。
主要功能包括 DeepSpeed 分布式训练、半精度训练、适配器模型支持和各种评估功能。

主要功能：
- 支持 DeepSpeed 分布式训练框架
- 半精度（FP16）训练支持
- 适配器模型（如 LoRA）集成
- GSM8K 数学推理评估
- 自定义优化器和学习率调度器
- FLOP 计数和性能监控

类：
- LLMTrainer: 主要的 LLM 训练器类，继承自 GeneralTorchTrainer

函数：
- call_llm_trainer: 训练器工厂函数
"""

import torch
import logging

# 尝试导入 DeepSpeed（用于分布式训练）
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except:
    deepspeed = None
    DeepSpeedEngine = None

# 导入 FederatedScope 核心模块
from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler

# 导入 LLM 相关模块
from federatedscope.llm.model.adapter_builder import AdapterModel
from federatedscope.llm.dataset.llm_dataset import DefaultToken   # 为 GSM8K 评估添加
from federatedscope.llm.misc.fschat import FSChatBot_My   # 为 GSM8K 评估添加
from federatedscope.llm.eval.eval_for_gsm8k.eval import *   # 为 GSM8K 评估添加
from federatedscope.llm.dataset.llm_dataset import PROMPT_DICT   # 为 GSM8K 评估添加

logger = logging.getLogger(__name__)


class LLMTrainer(GeneralTorchTrainer):
    """
    大型语言模型训练器类。
    
    继承自 GeneralTorchTrainer，专门用于大型语言模型的联邦学习训练。
    支持 DeepSpeed 分布式训练、半精度训练、适配器模型和各种评估功能。
    
    主要特性：
    - DeepSpeed 集成：支持大规模模型的分布式训练
    - 半精度训练：减少内存使用和加速训练
    - 适配器支持：集成 LoRA 等参数高效微调方法
    - 评估功能：支持 GSM8K 等下游任务评估
    - FLOP 计数：监控计算复杂度
    - 自定义钩子：针对 LLM 训练的特殊处理
    
    属性：
        cfg: 训练配置对象
        model: 语言模型实例
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 训练设备
    """
    
    def _hook_on_fit_start_numerical_precision(self, ctx):
        """
        在训练开始时设置数值精度的钩子函数。
        
        该方法在训练开始时被调用，用于配置半精度训练。
        如果启用了半精度训练且未使用 DeepSpeed，则将模型转换为半精度。
        
        参数:
            ctx: 训练上下文对象，包含模型、配置等信息
        """
        if self.cfg.train.is_enable_half:
            if not ctx.cfg.llm.deepspeed.use:
                ctx.model = ctx.model.half()  # 启用半精度训练

    def _hook_on_fit_start_init(self, ctx):
        """
        在训练开始时初始化的钩子函数。
        
        该方法负责初始化训练所需的组件，包括 DeepSpeed 引擎、优化器和学习率调度器。
        根据配置选择使用 DeepSpeed 分布式训练或标准训练模式。
        
        参数:
            ctx: 训练上下文对象，包含模型、配置等信息
        """
        if ctx.cfg.llm.deepspeed.use:
            # 使用 DeepSpeed 分布式训练
            # TODO: 保存 ctx.optimizer 和 ctx.scheduler
            # TODO: 客户端是否应该共享相同的 `ctx.model_engine`？
            assert deepspeed is not None, "Please install deepspeed."
            if not hasattr(ctx, 'model_engine'):
                # 初始化 DeepSpeed 引擎，包括模型、优化器和调度器
                ctx.model_engine, ctx.optimizer, _, ctx.scheduler = \
                    deepspeed.initialize(
                        config=ctx.cfg.llm.deepspeed.ds_config,
                        model=ctx.model,
                        model_parameters=filter(lambda p: p.requires_grad,
                                                ctx.model.parameters()),
                    )
            # 设置设备为 DeepSpeed 引擎的本地排名
            ctx.device = ctx.model_engine.local_rank
            # 检查是否启用了半精度训练
            if ctx.cfg.train.is_enable_half:
                ctx.fp16 = ctx.model_engine.fp16_enabled()
        else:
            # 标准训练模式：准备模型和优化器
            ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                # 在此处初始化优化器，避免在不同例程中重复使用优化器
                ctx.optimizer = get_optimizer(
                    ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                ctx.scheduler = get_scheduler(
                    ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

        # 准备统计信息变量
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)  # 批次总损失
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)  # 正则化总损失
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)  # 样本数量
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)  # 真实标签
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)  # 预测概率
        
        # 为 GSM8K 评估保存验证数据加载器的副本
        if not hasattr(ctx, 'val_loader_copy'):
            ctx.val_loader_copy = ctx.val_loader
        
    def _hook_on_batch_forward(self, ctx):
        """
        批次前向传播的钩子函数。
        
        该方法处理输入数据，执行模型前向传播，计算损失。
        支持 DeepSpeed 和标准训练模式，并处理 NaN 损失的情况。
        
        参数:
            ctx: 训练上下文对象，包含数据批次、模型等信息
        """
        # 将输入数据移动到指定设备
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

        if ctx.cfg.llm.deepspeed.use:
            # 使用 DeepSpeed 引擎进行前向传播
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)
        else:
            # 使用标准模型进行前向传播
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        # 提取模型输出
        logits = outputs.logits  # 模型输出的 logits
        loss = outputs.loss  # 计算的损失
        
        # 检查损失是否为 NaN
        if torch.isnan(loss):
            # 如果损失为 NaN，跳过此批次
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        # 保存批次结果到上下文
        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        """
        批次反向传播的钩子函数。
        
        该方法执行梯度计算、梯度裁剪和参数更新。
        支持 DeepSpeed 和标准训练模式的不同反向传播流程。
        
        参数:
            ctx: 训练上下文对象，包含模型、优化器等信息
        """
        # 如果当前批次被跳过，则直接返回
        if ctx.skip_this_batch:
            return

        if ctx.cfg.llm.deepspeed.use:
            # 使用 DeepSpeed 进行反向传播和参数更新
            ctx.model_engine.backward(ctx.loss_task)
            ctx.model_engine.step()
        else:
            # 标准的反向传播流程
            ctx.optimizer.zero_grad()  # 清零梯度
            ctx.loss_task.backward()  # 反向传播

            if ctx.grad_clip > 0:
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                               ctx.grad_clip)

            ctx.optimizer.step()  # 更新参数
        if ctx.scheduler is not None:
            ctx.scheduler.step()  # 更新学习率

    def _hook_on_batch_end(self, ctx):
        """
        批次结束时的钩子函数。
        Hook function at the end of each batch.
        
        更新统计信息，处理 NaN 损失的重试逻辑。
        Updates statistics and handles retry logic for NaN losses.
        
        参数:
            ctx: 训练上下文对象，包含批次信息和配置
        """
        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                # 在训练和微调模式下使用新数据重试
                # Retry with new data in train and finetune
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        # 更新统计信息
        ctx.num_samples += ctx.batch_size  # 累计样本数
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size  # 累计损失
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))  # 累计正则化损失

    def _hook_on_fit_end(self, ctx):
        """
        训练结束时的钩子函数。
        Hook function at the end of training.
        
        计算平均损失并返回结果字典。对于测试模式，还会进行 GSM8K 数据集的生成式评估。
        Computes average loss and returns result dictionary. For test mode, also performs generative evaluation on GSM8K dataset.
        
        参数:
            ctx: 训练上下文对象，包含模型、数据和配置信息
        """
        # 计算平均损失
        avg_loss = 0 if float(
            ctx.num_samples) == 0 else ctx.loss_batch_total / float(
                ctx.num_samples)
        # 构建评估结果字典
        eval_results = {
                f'{ctx.cur_split}_loss': ctx.loss_batch_total,  # 总损失
                f'{ctx.cur_split}_total': ctx.num_samples,  # 总样本数
                f'{ctx.cur_split}_avg_loss': avg_loss,  # 平均损失
        }
        
        # 为 GSM8K 数据集评估添加
        # added by me, evaluating on GSM8K dataset
        if ctx.cur_mode == MODE.TEST:
            # 创建聊天机器人进行生成式评估
            fschatbot = FSChatBot_My(ctx.model.cpu(), ctx.cfg)
            answers = []  # 存储答案正确性
            for batch in ctx.val_loader_copy:
                for instruction, _, output in zip(batch['instruction'], batch['input'], batch['output']):
                    # 构建提示文本
                    input_text = build_prompt(instruction, N_SHOT, COT_FLAG)
                    # 设置生成参数
                    generate_kwargs = dict(max_new_tokens=256, top_p=0.95, temperature=0.8)
                    # 生成模型回答
                    model_completion = fschatbot.generate(input_text, generate_kwargs)
                    # 清理并提取答案
                    model_answer = clean_answer(model_completion)
                    # 检查答案是否正确
                    is_cor = is_correct(model_answer, output)
                    answers.append(is_cor)
                    # 打印详细信息
                    print(f'Question: {instruction}\n\n'
                          f'Answers: {extract_answer_from_output(output)}\n\n'
                          f'Model Answers: {model_answer}\n\n'
                          f'Model Completion: {model_completion}\n\n'
                          f'Is correct: {is_cor}\n\n')

                    # 打印统计信息
                    print(f'Num of total question: {len(answers)}, '
                          f'correct num: {sum(answers)}, '
                          f'correct rate: {float(sum(answers))/len(answers)}.')
            # 计算准确率
            eval_results[f'{ctx.cur_split}_acc'] = float(sum(answers))/len(answers)
        
        # 设置评估指标
        setattr(ctx, 'eval_metrics', eval_results)
                
        # TODO: 将此功能作为钩子函数
        # TODO: make this as a hook function
        # 将可训练部分移动到 CPU，可以节省内存但会消耗时间
        # Move trainable part to `cpu`, which can save memory but cost time
        if ctx.cfg.llm.adapter.mv_to_cpu:
            for p in ctx.model.parameters():
                if p.requires_grad:
                    p.data = p.to('cpu')  # 将参数移动到 CPU
                    if p.grad is not None:
                        p.grad.data = p.grad.to('cpu')  # 将梯度移动到 CPU

    def _hook_on_batch_forward_flop_count(self, ctx):
        """
        计算联邦学习过程中 FLOPs 的监控钩子函数。
        The monitoring hook to calculate the flops during the fl course

        该方法用于监控和统计模型前向传播过程中的浮点运算次数（FLOPs），
        支持标准模型和适配器模型的 FLOPs 计算。
        
        注意：
        对于前向过程不仅基于 ctx.model 的自定义情况，请重写此函数（继承情况）或替换此钩子（插件情况）。
        Note:
          For customized cases that the forward process is not only \
          based on ctx.model, please override this function (inheritance \
          case) or replace this hook (plug-in case)

          修改的属性和相应操作如下所示：
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            属性 Attribute                      操作 Operation
            ==================================  ===========================
            ``ctx.monitor``                     跟踪平均 FLOPs Track average flops
            ==================================  ===========================
            
        参数:
            ctx: 训练上下文对象，包含模型、数据批次和监控器
        """

        # 如果垃圾回收不及时触发，该过程可能会占用大量显存
        # 当有大量显存剩余时。设置 `eval.count_flops = False` 来避免这种情况。
        # The process may occupy a large amount of video memory
        # if the garbage collection is not triggered in time
        # when there is plenty of video memory left. Set
        # `eval.count_flops = False` to avoid this.
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Please check whether this is you want.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            # 计算每个样本的 FLOPs
            # calculate the flops_per_sample
            try:
                # 将数据移动到设备
                input_ids = ctx.data_batch['input_ids'].to(ctx.device)
                labels = ctx.data_batch['labels'].to(ctx.device)
                attention_mask = ctx.data_batch['attention_mask'].to(
                    ctx.device)
                from fvcore.nn import FlopCountAnalysis
                if isinstance(ctx.model, AdapterModel):
                    # 对于适配器模型，分析底层模型的 FLOPs
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model.model,
                        inputs=(input_ids, attention_mask)).total()
                else:
                    # 对于标准模型，直接分析模型的 FLOPs
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model, inputs=(input_ids, attention_mask)).total()
                # 跟踪平均 FLOPs
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except Exception as e:
                logger.warning("When using count flops functions, torch's "
                               "garbage collection mechanism may not be "
                               "timely resulting in OOM, please set "
                               "`cfg.eval.count_flops` to `False` "
                               "to avoid error or warning like this.")
                logger.error(e)
                # 在第一次失败时发出警告
                # Raise warning at the first failure
                logger.warning(
                    "current flop count implementation is for general LLM "
                    "trainer case: "
                    "1) ctx.data_batch contains [input_ids, labels, "
                    "attn_mask]; and 2) the ctx.model takes first two "
                    "arguments should be and attention_mask. "
                    "If ctx.model is an adapter model, the model in 2) has "
                    "been replaced by ctx.model.model. "
                    "Please check the forward format or implement your own "
                    "flop_count function")
                ctx.monitor.flops_per_sample = -1

        # 默认情况下，我们假设数据具有相同的输入形状，
        # 因此简单地乘以 FLOPs 以避免冗余的前向传播
        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * \
            ctx.batch_size


def call_llm_trainer(trainer_type):
    """
    LLM 训练器调用函数。
    LLM trainer calling function.
    
    根据训练器类型返回相应的训练器构建器。
    Returns the corresponding trainer builder based on trainer type.
    
    Args:
        trainer_type (str): 训练器类型 / Trainer type
        
    Returns:
        class: 训练器构建器类 / Trainer builder class
    """
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder


# 注册 LLM 训练器
# Register LLM trainer
register_trainer('llmtrainer', call_llm_trainer)
