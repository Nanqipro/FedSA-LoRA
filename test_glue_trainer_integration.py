#!/usr/bin/env python3
"""
Test script to verify LoRA monitoring integration with GLUETrainer
"""

import torch
import logging
import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/home/nanqipro01/gitlocal/FedSA-LoRA')

from federatedscope.glue.trainer.trainer import GLUETrainer
from federatedscope.lora_matrix_monitor import LoRAMatrixMonitor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        self.device = 'cpu'
        self.backend = 'torch'
        self.federate = type('obj', (object,), {
            'client_num': 3,
            'process_num': 1,
            'mode': 'standalone'
        })()
        self.model = type('obj', (object,), {
            'type': 'huggingface_llm',
            'model_name_or_path': 'FacebookAI/roberta-large'
        })()
        self.data = type('obj', (object,), {'type': 'mnli@glue'})()
        self.train = type('obj', (object,), {
            'local_update_steps': 10,
            'batch_size': 16
        })()
        self.eval = type('obj', (object,), {
            'freq': 1,
            'metrics': ['accuracy'],
            'best_res_update_round_wise_key': 'test_accuracy',
            'count_flops': False,
            'monitoring': []
        })()
        self.criterion = type('obj', (object,), {'type': 'CrossEntropyLoss'})()
        self.optimizer = type('obj', (object,), {'type': 'AdamW', 'lr': 0.02})()
        self.scheduler = type('obj', (object,), {'type': 'linear'})()
        self.grad = type('obj', (object,), {'grad_clip': 1.0})()
        self.regularizer = type('obj', (object,), {'type': ''})()
        self.seed = 12345
        self.use_gpu = False
        self.early_stop = type('obj', (object,), {'patience': 5})()
        self.wandb = type('obj', (object,), {'use': False})()
        self.tb = type('obj', (object,), {'use': False})()
        self.outdir = '/tmp/test_output'
        self.finetune = type('obj', (object,), {'before_eval': False})()

class MockModel(torch.nn.Module):
    """Mock model with LoRA parameters for testing"""
    def __init__(self):
        super().__init__()
        # Simulate the base_model structure from PEFT
        self.base_model = torch.nn.Module()
        self.base_model.model = torch.nn.Module()
        self.base_model.model.roberta = torch.nn.Module()
        self.base_model.model.roberta.encoder = torch.nn.Module()
        self.base_model.model.roberta.encoder.layer = torch.nn.ModuleList()
        
        # Add two layers with LoRA parameters
        for i in range(2):
            layer = torch.nn.Module()
            layer.attention = torch.nn.Module()
            layer.attention.self = torch.nn.Module()
            layer.attention.self.query = torch.nn.Module()
            layer.attention.self.value = torch.nn.Module()
            
            # Add LoRA A and B matrices with correct naming
            layer.attention.self.query.lora_A = torch.nn.Module()
            layer.attention.self.query.lora_A.default = torch.nn.Module()
            layer.attention.self.query.lora_A.default.weight = torch.nn.Parameter(torch.randn(8, 768))
            
            layer.attention.self.query.lora_B = torch.nn.Module()
            layer.attention.self.query.lora_B.default = torch.nn.Module()
            layer.attention.self.query.lora_B.default.weight = torch.nn.Parameter(torch.randn(768, 8))
            
            layer.attention.self.value.lora_A = torch.nn.Module()
            layer.attention.self.value.lora_A.default = torch.nn.Module()
            layer.attention.self.value.lora_A.default.weight = torch.nn.Parameter(torch.randn(8, 768))
            
            layer.attention.self.value.lora_B = torch.nn.Module()
            layer.attention.self.value.lora_B.default = torch.nn.Module()
            layer.attention.self.value.lora_B.default.weight = torch.nn.Parameter(torch.randn(768, 8))
            
            self.base_model.model.roberta.encoder.layer.append(layer)
    
    def forward(self, x):
        return x

def create_mock_data():
    """Create mock data in dictionary format"""
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(10, 768),  # input_ids
            torch.randint(0, 2, (10,))  # labels
        ),
        batch_size=2
    )
    return {
        'train': train_loader,
        'test': train_loader,
        'val': train_loader
    }

def test_glue_trainer_lora_monitoring():
    """Test LoRA monitoring integration with GLUETrainer"""
    logger.info("Starting GLUETrainer LoRA monitoring integration test")
    
    # Create mock objects
    config = MockConfig()
    model = MockModel()
    data = create_mock_data()
    device = torch.device('cpu')
    
    try:
        # Create a mock monitor
        from federatedscope.core.monitors.monitor import Monitor
        monitor = Monitor(config)
        
        # Initialize GLUETrainer
        logger.info("Initializing GLUETrainer...")
        trainer = GLUETrainer(
            model=model,
            data=data,
            device=device,
            config=config,
            only_for_eval=False,
            monitor=monitor
        )
        
        # Check if LoRA monitor was initialized
        if trainer.lora_monitor is not None:
            logger.info("✓ LoRA monitor successfully initialized in GLUETrainer")
            
            # Test the monitoring hooks
            logger.info("Testing LoRA monitoring hooks...")
            
            # Create a mock context
            class MockContext:
                def __init__(self, model):
                    self.model = model
                    self.eval_metrics = {}
            
            ctx = MockContext(model)
            
            # Test start monitoring hook
            trainer._hook_on_fit_start_lora_monitor(ctx)
            logger.info("✓ Start monitoring hook executed successfully")
            
            # Test end monitoring hook
            trainer._hook_on_fit_end_lora_monitor(ctx)
            logger.info("✓ End monitoring hook executed successfully")
            
            # Check if log file was created
            if os.path.exists('lora_matrix_sizes.txt'):
                logger.info("✓ LoRA monitoring log file created successfully")
                with open('lora_matrix_sizes.txt', 'r') as f:
                    content = f.read()
                    logger.info(f"Log file content:\n{content}")
            else:
                logger.warning("⚠ LoRA monitoring log file not found")
            
        else:
            logger.error("✗ LoRA monitor was not initialized in GLUETrainer")
            return False
        
        logger.info("✓ All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_glue_trainer_lora_monitoring()
    if success:
        print("\n🎉 GLUETrainer LoRA monitoring integration test PASSED!")
    else:
        print("\n❌ GLUETrainer LoRA monitoring integration test FAILED!")
        sys.exit(1)