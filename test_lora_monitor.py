#!/usr/bin/env python3
"""
测试LoRA矩阵监控功能
"""

import sys
import os
sys.path.append('.')

try:
    from federatedscope.lora_matrix_monitor import LoRAMatrixMonitor
    print("✓ 成功导入LoRAMatrixMonitor")
except ImportError as e:
    print(f"✗ 导入LoRAMatrixMonitor失败: {e}")
    sys.exit(1)

# 测试监控器初始化
try:
    monitor = LoRAMatrixMonitor(log_file_path="test_lora_monitor.txt")
    print("✓ 成功初始化LoRAMatrixMonitor")
except Exception as e:
    print(f"✗ 初始化LoRAMatrixMonitor失败: {e}")
    sys.exit(1)

# 创建一个简单的模拟模型来测试
import torch
import torch.nn as nn

class MockLoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 模拟LoRA参数
        self.base_model = nn.ModuleDict({
            'model': nn.ModuleDict({
                'roberta': nn.ModuleDict({
                    'encoder': nn.ModuleDict({
                        'layer': nn.ModuleList([
                            nn.ModuleDict({
                                'attention': nn.ModuleDict({
                                    'self': nn.ModuleDict({
                                        'query': nn.ModuleDict({
                                            'lora_A': nn.Linear(768, 8, bias=False),
                                            'lora_B': nn.Linear(8, 768, bias=False)
                                        })
                                    })
                                })
                            }) for _ in range(2)
                        ])
                    })
                })
            })
        })
        
    def named_parameters(self):
        for name, param in super().named_parameters():
            yield name, param

# 测试监控功能
try:
    model = MockLoRAModel()
    print("✓ 成功创建模拟模型")
    
    # 测试矩阵信息提取
    matrix_info = monitor.extract_matrix_info(model)
    print(f"✓ 成功提取矩阵信息: {len(matrix_info['lora_A'])} A矩阵, {len(matrix_info['lora_B'])} B矩阵")
    
    # 测试监控开始
    stats = monitor.start_monitoring(model)
    print("✓ 成功开始监控")
    
    # 测试获取当前统计
    current_stats = monitor.get_current_stats(model)
    if current_stats:
        print(f"✓ 成功获取当前统计: {current_stats}")
    
    # 测试记录矩阵大小
    monitor.log_matrix_sizes(model, round_num=1, client_id="test_client")
    print("✓ 成功记录矩阵大小")
    
    # 测试结束监控
    end_stats = monitor.end_monitoring()
    print(f"✓ 成功结束监控: {end_stats}")
    
    print("\n所有测试通过！监控功能正常工作。")
    
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n检查生成的日志文件:")
if os.path.exists("test_lora_monitor.txt"):
    with open("test_lora_monitor.txt", 'r', encoding='utf-8') as f:
        content = f.read()
        print("日志文件内容:")
        print("-" * 40)
        print(content)
        print("-" * 40)
else:
    print("✗ 日志文件未生成")