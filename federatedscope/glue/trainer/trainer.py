# 导入必要的库
import torch
import logging
# 尝试导入 DeepSpeed，用于分布式训练和模型优化
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except:
    deepspeed = None
    DeepSpeedEngine = None
# 导入 FederatedScope 相关模块
from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.glue.model.adapter_builder import AdapterModel
# 导入 Hugging Face datasets 库中的评估指标
from datasets import load_metric
import numpy as np

# 获取日志记录器
logger = logging.getLogger(__name__)


class GLUETrainer(GeneralTorchTrainer):
    """
    GLUE 任务的训练器类，继承自 GeneralTorchTrainer
    专门用于处理 GLUE 基准测试任务的训练和评估
    """
    
    def _hook_on_fit_start_numerical_precision(self, ctx):
        """
        在训练开始时设置数值精度的钩子函数
        如果启用半精度训练且未使用 DeepSpeed，则将模型转换为半精度
        
        Args:
            ctx: 训练上下文，包含模型、配置等信息
        """
        if self.cfg.train.is_enable_half:
            if not ctx.cfg.llm.deepspeed.use:
                ctx.model = ctx.model.half()

    def _hook_on_fit_start_init(self, ctx):
        """
        在训练开始时进行初始化的钩子函数
        设置模型、优化器、调度器等训练所需组件
        
        Args:
            ctx: 训练上下文，包含模型、配置等信息
        """
        if ctx.cfg.llm.deepspeed.use:
            # Enable deepspeed
            # TODO: save ctx.optimizer and ctx.scheduler
            # TODO: should clients share the same `ctx.model_engine`?
            assert deepspeed is not None, "Please install deepspeed."
            if not hasattr(ctx, 'model_engine'):
                ctx.model_engine, ctx.optimizer, _, ctx.scheduler = \
                    deepspeed.initialize(
                        config=ctx.cfg.llm.deepspeed.ds_config,
                        model=ctx.model,
                        model_parameters=filter(lambda p: p.requires_grad,
                                                ctx.model.parameters()),
                    )
            # Enable all cards from 0
            ctx.device = ctx.model_engine.local_rank
            if ctx.cfg.train.is_enable_half:
                ctx.fp16 = ctx.model_engine.fp16_enabled()
        else:
            # prepare model and optimizer
            ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                # 在此处初始化优化器，避免在不同例程中重复使用优化器
                if ctx.cfg.llm.adapter.args[0].get('adapter_method', '') == "vera":
                    # 为 VeRA 适配器引入分离的学习率：分类头和适配层使用不同的学习率
                    vera_params = [param for name, param in ctx.model.named_parameters() if "vera" in name and param.requires_grad]
                    other_params = [param for name, param in ctx.model.named_parameters() if "vera" not in name and param.requires_grad]
                    optimizer_grouped_parameters = [
                        {'params': vera_params, 'lr': ctx.cfg.train.optimizer.lr},  # VeRA 参数使用标准学习率
                        {'params': other_params, 'lr': ctx.cfg.train.vera.lr_c}     # 其他参数使用分类头学习率
                    ]
                    from transformers import AdamW, get_linear_schedule_with_warmup
                    ctx.optimizer = AdamW(optimizer_grouped_parameters, no_deprecation_warning=True)
                    # 设置线性预热调度器
                    ctx.scheduler = get_linear_schedule_with_warmup(
                                    ctx.optimizer, 
                                    num_warmup_steps=0.06 * ctx.cfg.train.local_update_steps * ctx.cfg.federate.total_round_num, 
                                    num_training_steps=ctx.cfg.train.local_update_steps * ctx.cfg.federate.total_round_num
                    )
                else:
                    # 使用标准的优化器和调度器构建器
                    ctx.optimizer = get_optimizer(
                        ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                    ctx.scheduler = get_scheduler(
                        ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

        # 准备统计变量
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)    # 批次损失总和
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)  # 正则化损失总和
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)          # 样本总数
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)             # 真实标签列表
        ctx.ys_pred = CtxVar([], LIFECYCLE.ROUTINE)             # 预测标签列表（为 GLUE 任务修改）

    def _hook_on_batch_forward(self, ctx):
        """
        批次前向传播的钩子函数
        处理输入数据，执行模型前向传播，计算损失和预测结果
        
        Args:
            ctx: 训练上下文，包含数据批次、模型等信息
        """
        # 将输入数据移动到指定设备（GPU/CPU）
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['label'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
        
        # 根据是否使用 DeepSpeed 选择不同的模型调用方式
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

        # 获取预测结果（为 GLUE 任务修改）
        preds = outputs.logits.argmax(dim=-1)
        loss = outputs.loss
        
        # 检查损失是否为 NaN，如果是则跳过此批次
        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        # 保存批次级别的结果
        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)      # 真实标签
        ctx.y_pred = CtxVar(preds, LIFECYCLE.BATCH)       # 预测标签
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)    # 批次损失
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)  # 批次大小

    def _hook_on_batch_backward(self, ctx):
        """
        批次反向传播的钩子函数
        执行梯度计算、梯度裁剪和参数更新
        
        Args:
            ctx: 训练上下文，包含模型、优化器等信息
        """
        # 如果跳过此批次，则直接返回
        if ctx.skip_this_batch:
            return

        # 根据是否使用 DeepSpeed 选择不同的反向传播方式
        if ctx.cfg.llm.deepspeed.use:
            # 使用 DeepSpeed 引擎进行反向传播和参数更新
            ctx.model_engine.backward(ctx.loss_task)
            ctx.model_engine.step()
        else:
            # 使用标准的 PyTorch 训练流程
            ctx.optimizer.zero_grad()  # 清零梯度
            ctx.loss_task.backward()   # 反向传播计算梯度

            # 如果设置了梯度裁剪，则进行梯度裁剪
            if ctx.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                               ctx.grad_clip)

            ctx.optimizer.step()  # 更新模型参数
        
        # 如果有学习率调度器，则更新学习率
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def _hook_on_batch_end(self, ctx):
        """
        批次结束时的钩子函数
        更新统计信息，处理 NaN 损失的重试逻辑
        
        Args:
            ctx: 训练上下文，包含批次结果和统计信息
        """
        # 如果跳过此批次，处理重试逻辑
        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                # 在训练和微调模式下使用新数据重试
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return
        
        # 更新统计信息
        ctx.num_samples += ctx.batch_size  # 累计样本数量
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size  # 累计加权损失
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))    # 累计正则化损失
        
        # 缓存标签用于评估，使用 extend 而不是 append
        ctx.ys_true.extend(ctx.y_true.detach().cpu().numpy())  # 真实标签
        ctx.ys_pred.extend(ctx.y_pred.detach().cpu().numpy())  # 预测标签
        
    def _hook_on_fit_end(self, ctx):
        """
        训练/评估结束时的钩子函数
        计算平均损失和 GLUE 评估指标，可选择将可训练参数移至 CPU
        
        Args:
            ctx: 训练上下文，包含累计的统计信息
        """
        # 计算平均损失
        avg_loss = 0 if float(
            ctx.num_samples) == 0 else ctx.loss_batch_total / float(
                ctx.num_samples)
        
        # 构建基础评估结果
        eval_results = {
                f'{ctx.cur_split}_loss': ctx.loss_batch_total,      # 总损失
                f'{ctx.cur_split}_total': ctx.num_samples,          # 总样本数
                f'{ctx.cur_split}_avg_loss': avg_loss               # 平均损失
        }
        
        # 计算 GLUE 特定的评估指标
        glue_metric = load_metric('glue', ctx.cfg.data.type.split('@')[0], trust_remote_code=True)
        eval_metric = glue_metric.compute(predictions=ctx.ys_pred, references=ctx.ys_true)
        # 将 GLUE 指标添加到评估结果中
        for k, v in eval_metric.items():
            eval_results[f'{ctx.cur_split}_{k}'] = v
        
        # 设置评估指标到上下文
        setattr(ctx, 'eval_metrics', eval_results)
        
        # TODO: 将此功能实现为钩子函数
        # 将可训练参数移至 CPU，可以节省内存但会消耗时间
        if ctx.cfg.llm.adapter.mv_to_cpu:
            for p in ctx.model.parameters():
                if p.requires_grad:
                    p.data = p.to('cpu')
                    if p.grad is not None:
                        p.grad.data = p.grad.to('cpu')

    def _hook_on_batch_forward_flop_count(self, ctx):
        """
        用于在联邦学习过程中计算 FLOP（浮点运算次数）的监控钩子函数

        注意:
          对于前向传播过程不仅基于 ctx.model 的自定义情况，
          请重写此函数（继承情况）或替换此钩子（插件情况）

          修改的属性和相应操作如下所示:
            ==================================  ===========================
            属性                                操作
            ==================================  ===========================
            ``ctx.monitor``                     跟踪平均 FLOP 数
            ==================================  ===========================
        """

        # 当显存充足时，如果垃圾回收不及时触发，
        # 此过程可能会占用大量显存。设置 `eval.count_flops = False` 来避免此问题。
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"训练器 {type(self)} 不包含有效的监控器，"
                f"这可能是由于在初始化训练器子类时没有传递有效的监控器实例造成的。"
                f"请检查这是否是您想要的。")
            return

        # 如果启用 FLOP 计数且每样本 FLOP 数为 0，则计算 FLOP 数
        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            # 计算每样本的 FLOP 数
            try:
                input_ids = ctx.data_batch['input_ids'].to(ctx.device)
                attention_mask = ctx.data_batch['attention_mask'].to(
                    ctx.device)
                from fvcore.nn import FlopCountAnalysis
                # 根据模型类型选择不同的 FLOP 分析方式
                if isinstance(ctx.model, AdapterModel):
                    # 对于适配器模型，分析底层模型
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model.model,
                        inputs=(input_ids, attention_mask)).total()
                else:
                    # 对于标准模型，直接分析
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model, inputs=(input_ids, attention_mask)).total()
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except Exception as e:
                logger.warning("使用 FLOP 计数功能时，torch 的垃圾回收机制可能不及时，"
                               "导致 OOM，请将 `cfg.eval.count_flops` 设置为 `False` "
                               "以避免此类错误或警告。")
                logger.error(e)
                # 在首次失败时发出警告
                logger.warning(
                    "当前的 FLOP 计数实现适用于通用 LLM 训练器情况："
                    "1) ctx.data_batch 包含 [input_ids, labels, attn_mask]；"
                    "2) ctx.model 的前两个参数应该是 input_ids 和 attention_mask。"
                    "如果 ctx.model 是适配器模型，则 2) 中的模型已被 ctx.model.model 替换。"
                    "请检查前向传播格式或实现您自己的 flop_count 函数")
                ctx.monitor.flops_per_sample = -1

        # 默认情况下，我们假设数据具有相同的输入形状，
        # 因此简单地乘以 FLOP 数以避免冗余的前向传播
        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * \
            ctx.batch_size


def call_glue_trainer(trainer_type):
    """
    GLUE 训练器的工厂函数
    根据训练器类型返回相应的训练器构建器
    
    Args:
        trainer_type (str): 训练器类型，应为 'gluetrainer'
    
    Returns:
        GLUETrainer: GLUE 训练器类
    """
    if trainer_type == 'gluetrainer':
        trainer_builder = GLUETrainer
        return trainer_builder


# 注册 GLUE 训练器到 FederatedScope 框架
register_trainer('gluetrainer', call_glue_trainer)
