#!/usr/bin/env python3
"""
LoRA Matrix Monitor Script
监控LoRA A和B矩阵在每轮训练中的大小变化情况
"""

import torch
import os
import datetime
from typing import Dict, List, Tuple
import json


class LoRAMatrixMonitor:
    """LoRA矩阵监控器"""
    
    def __init__(self, log_file_path: str = "lora_matrix_sizes.txt"):
        """
        初始化监控器
        
        Args:
            log_file_path: 日志文件路径
        """
        self.log_file_path = log_file_path
        self.round_count = 0
        self.monitoring_active = False
        self.initial_stats = {}
        
        # 创建日志文件并写入头部信息
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LoRA Matrix Size Monitor Log\n")
            f.write(f"Started at: {datetime.datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
    
    def extract_matrix_info(self, model) -> Dict[str, Dict[str, Tuple[int, ...]]]:
        """
        提取模型中LoRA矩阵的信息
        
        Args:
            model: PyTorch模型
            
        Returns:
            包含矩阵信息的字典
        """
        matrix_info = {
            'lora_A': {},
            'lora_B': {}
        }
        
        for name, param in model.named_parameters():
            if 'lora_A' in name:
                # 提取层信息
                layer_info = self._extract_layer_info(name)
                matrix_info['lora_A'][layer_info] = {
                    'shape': tuple(param.shape),
                    'size': param.numel(),
                    'dtype': str(param.dtype),
                    'requires_grad': param.requires_grad
                }
            elif 'lora_B' in name:
                # 提取层信息
                layer_info = self._extract_layer_info(name)
                matrix_info['lora_B'][layer_info] = {
                    'shape': tuple(param.shape),
                    'size': param.numel(),
                    'dtype': str(param.dtype),
                    'requires_grad': param.requires_grad
                }
        
        return matrix_info
    
    def _extract_layer_info(self, param_name: str) -> str:
        """
        从参数名称中提取层信息
        
        Args:
            param_name: 参数名称
            
        Returns:
            层信息字符串
        """
        # 例如: base_model.model.roberta.encoder.layer.0.attention.self.query.lora_A.default.weight
        # 提取: layer.0.attention.self.query
        parts = param_name.split('.')
        
        # 找到layer的位置
        layer_idx = -1
        for i, part in enumerate(parts):
            if part == 'layer' and i + 1 < len(parts):
                layer_idx = i
                break
        
        if layer_idx != -1:
            # 提取从layer到lora_A/lora_B之前的部分
            end_idx = -1
            for i in range(layer_idx, len(parts)):
                if 'lora_A' in parts[i] or 'lora_B' in parts[i]:
                    end_idx = i
                    break
            
            if end_idx != -1:
                return '.'.join(parts[layer_idx:end_idx])
        
        return param_name
    
    def start_monitoring(self, model):
        """开始监控LoRA矩阵"""
        self.monitoring_active = True
        self.initial_stats = self.extract_matrix_info(model)
        self.log_matrix_sizes(model, round_num=0)
        return self.initial_stats
    
    def end_monitoring(self):
        """结束监控并返回统计信息"""
        self.monitoring_active = False
        return {
            'total_rounds_monitored': self.round_count,
            'monitoring_completed': True
        }
    
    def get_current_stats(self, model):
        """获取当前LoRA矩阵统计信息"""
        if not self.monitoring_active:
            return None
        
        matrix_info = self.extract_matrix_info(model)
        if not matrix_info:
            return None
        
        total_a_params = sum(info['size'] for info in matrix_info['lora_A'].values())
        total_b_params = sum(info['size'] for info in matrix_info['lora_B'].values())
        
        return {
            'lora_A_total_params': total_a_params,
            'lora_B_total_params': total_b_params,
            'lora_total_params': total_a_params + total_b_params,
            'num_lora_layers': len(matrix_info['lora_A']) + len(matrix_info['lora_B'])
        }

    def log_matrix_sizes(self, model, round_num: int = None, client_id: str = "unknown"):
        """
        记录当前轮次的矩阵大小
        
        Args:
            model: PyTorch模型
            round_num: 轮次编号
            client_id: 客户端ID
        """
        if round_num is None:
            round_num = self.round_count
            self.round_count += 1
        
        matrix_info = self.extract_matrix_info(model)
        
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"Round {round_num} - Client {client_id} - {datetime.datetime.now()}\n")
            f.write("-" * 60 + "\n")
            
            # 记录LoRA A矩阵信息
            f.write("LoRA A Matrices:\n")
            total_a_params = 0
            for layer_name, info in matrix_info['lora_A'].items():
                f.write(f"  {layer_name}:\n")
                f.write(f"    Shape: {info['shape']}\n")
                f.write(f"    Size: {info['size']} parameters\n")
                f.write(f"    Dtype: {info['dtype']}\n")
                f.write(f"    Requires Grad: {info['requires_grad']}\n")
                total_a_params += info['size']
            
            f.write(f"  Total LoRA A Parameters: {total_a_params}\n\n")
            
            # 记录LoRA B矩阵信息
            f.write("LoRA B Matrices:\n")
            total_b_params = 0
            for layer_name, info in matrix_info['lora_B'].items():
                f.write(f"  {layer_name}:\n")
                f.write(f"    Shape: {info['shape']}\n")
                f.write(f"    Size: {info['size']} parameters\n")
                f.write(f"    Dtype: {info['dtype']}\n")
                f.write(f"    Requires Grad: {info['requires_grad']}\n")
                total_b_params += info['size']
            
            f.write(f"  Total LoRA B Parameters: {total_b_params}\n")
            f.write(f"  Total LoRA Parameters (A+B): {total_a_params + total_b_params}\n")
            f.write("\n" + "=" * 80 + "\n\n")
    
    def get_matrix_summary(self, model) -> Dict:
        """
        获取矩阵摘要信息
        
        Args:
            model: PyTorch模型
            
        Returns:
            摘要信息字典
        """
        matrix_info = self.extract_matrix_info(model)
        
        summary = {
            'lora_A_count': len(matrix_info['lora_A']),
            'lora_B_count': len(matrix_info['lora_B']),
            'total_lora_A_params': sum(info['size'] for info in matrix_info['lora_A'].values()),
            'total_lora_B_params': sum(info['size'] for info in matrix_info['lora_B'].values()),
        }
        
        summary['total_lora_params'] = summary['total_lora_A_params'] + summary['total_lora_B_params']
        
        return summary


def monitor_model_matrices(model, round_num: int = None, client_id: str = "unknown", 
                          log_file: str = "lora_matrix_sizes.txt"):
    """
    便捷函数：监控模型中的LoRA矩阵
    
    Args:
        model: PyTorch模型
        round_num: 轮次编号
        client_id: 客户端ID
        log_file: 日志文件路径
    """
    monitor = LoRAMatrixMonitor(log_file)
    monitor.log_matrix_sizes(model, round_num, client_id)
    return monitor.get_matrix_summary(model)


if __name__ == "__main__":
    # 测试代码
    print("LoRA Matrix Monitor Script")
    print("This script is designed to be imported and used in training code.")
    print("Example usage:")
    print("  from lora_matrix_monitor import monitor_model_matrices")
    print("  summary = monitor_model_matrices(model, round_num=1, client_id='client_1')")
    print("  print(summary)")