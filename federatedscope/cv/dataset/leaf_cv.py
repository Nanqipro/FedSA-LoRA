"""LEAF 计算机视觉数据集模块

该模块实现了 LEAF 基准测试中的计算机视觉数据集，包括 FEMNIST 和 CelebA 数据集。
LEAF (Learning in Federated Environments) 是联邦学习的标准基准测试套件。

主要功能：
- 支持 FEMNIST 手写字符识别数据集
- 支持 CelebA 名人面部属性数据集
- 自动下载和预处理数据
- 按客户端分割数据
- 支持训练/验证/测试集划分
- 图像数据变换和增强

数据集特性：
- FEMNIST: 28x28 灰度图像，手写字符识别
- CelebA: 84x84x3 彩色图像，名人面部属性

参考文献：
    "LEAF: A Benchmark for Federated Settings"
    https://leaf.cmu.edu
"""

import os
import random
import json
import torch
import math

import numpy as np
import os.path as osp

from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from federatedscope.core.data.utils import save_local_data, download_url
from federatedscope.cv.dataset.leaf import LEAF

# 不同数据集的图像尺寸配置
IMAGE_SIZE = {'femnist': (28, 28), 'celeba': (84, 84, 3)}
# 不同数据集的图像模式配置
MODE = {'femnist': 'L', 'celeba': 'RGB'}


class LEAF_CV(LEAF):
    """LEAF 计算机视觉数据集类
    
    实现了 LEAF 基准测试中的计算机视觉数据集，支持 FEMNIST 和 CelebA 数据集。
    该类继承自 LEAF 基类，提供了联邦学习场景下的数据加载和预处理功能。
    
    Args:
        root (str): 数据集根目录路径
        name (str): 数据集名称，支持 'femnist' 或 'celeba'
        s_frac (float): 使用的数据集比例，默认为 0.3
        tr_frac (float): 每个任务的训练集比例，默认为 0.8
        val_frac (float): 每个任务的验证集比例，默认为 0.0
        train_tasks_frac (float): 训练任务的比例，默认为 1.0
        seed (int): 随机种子，用于数据分割的可重复性
        transform: 输入数据的变换函数
        target_transform: 标签数据的变换函数
    
    Attributes:
        data_dict (dict): 存储各客户端数据的字典
        s_frac (float): 数据集采样比例
        tr_frac (float): 训练集比例
        val_frac (float): 验证集比例
        seed (int): 随机种子
    
    Note:
        - 数据集会自动下载到指定的根目录
        - 支持按客户端分割数据，每个客户端对应一个任务
        - 图像会根据数据集类型自动调整尺寸和模式
    
    Reference:
        "LEAF: A Benchmark for Federated Settings"
        https://leaf.cmu.edu
    """
    def __init__(self,
                 root,
                 name,
                 s_frac=0.3,
                 tr_frac=0.8,
                 val_frac=0.0,
                 train_tasks_frac=1.0,
                 seed=123,
                 transform=None,
                 target_transform=None):
        # 保存数据集配置参数
        self.s_frac = s_frac                    # 数据集采样比例
        self.tr_frac = tr_frac                  # 训练集比例
        self.val_frac = val_frac                # 验证集比例
        self.seed = seed                        # 随机种子
        self.train_tasks_frac = train_tasks_frac # 训练任务比例
        
        # 调用父类构造函数
        super(LEAF_CV, self).__init__(root, name, transform, target_transform)
        # 加载已处理的数据文件
        files = os.listdir(self.processed_dir)
        files = [f for f in files if f.startswith('task_')]
        if len(files):
            # 按任务索引排序
            files.sort(key=lambda k: int(k[5:]))

            # 遍历每个任务文件，加载训练和测试数据
            for file in files:
                train_data, train_targets = torch.load(
                    osp.join(self.processed_dir, file, 'train.pt'))
                test_data, test_targets = torch.load(
                    osp.join(self.processed_dir, file, 'test.pt'))
                
                # 存储训练和测试数据
                self.data_dict[int(file[5:])] = {
                    'train': (train_data, train_targets),
                    'test': (test_data, test_targets)
                }
                
                # 如果存在验证集，也加载验证数据
                if osp.exists(osp.join(self.processed_dir, file, 'val.pt')):
                    val_data, val_targets = torch.load(
                        osp.join(self.processed_dir, file, 'val.pt'))
                    self.data_dict[int(file[5:])]['val'] = (val_data,
                                                            val_targets)
        else:
             raise RuntimeError(
                 'Please delete \'processed\' folder and try again!')

    @property
    def raw_file_names(self):
        """获取原始数据文件名列表
        
        Returns:
            list: 原始数据文件名列表，格式为 ['{dataset_name}_all_data.zip']
        """
        names = [f'{self.name}_all_data.zip']
        return names

    def download(self):
        """下载原始数据集文件
        
        从指定的 URL 下载数据集压缩文件到原始数据目录。
        如果文件已存在，则跳过下载。
        """
        # 从阿里云 OSS 下载数据集文件
        url = 'https://federatedscope.oss-cn-beijing.aliyuncs.com'
        os.makedirs(self.raw_dir, exist_ok=True)
        for name in self.raw_file_names:
            download_url(f'{url}/{name}', self.raw_dir)

    def __getitem__(self, index):
        """获取指定索引的客户端数据
        
        根据客户端索引返回该客户端的训练、测试和验证数据。
        图像数据会根据数据集类型自动调整尺寸和模式。
        
        Args:
            index (int): 客户端索引
        
        Returns:
            dict: 客户端数据字典，格式为：
                {
                    'train': [(image, target), ...],
                    'test': [(image, target), ...],
                    'val': [(image, target), ...]  # 如果存在验证集
                }
                其中 target 是目标类别标签
        
        Note:
            - 图像会转换为 PIL Image 对象
            - 会应用指定的数据变换函数
            - FEMNIST: 28x28 灰度图像
            - CelebA: 84x84x3 彩色图像
        """
        img_dict = {}
        data = self.data_dict[index]
        
        # 遍历训练、测试、验证集
        for key in data:
            img_dict[key] = []
            imgs, targets = data[key]
            
            # 处理每个样本
            for idx in range(targets.shape[0]):
                # 调整图像尺寸并转换为指定格式
                img = np.resize(imgs[idx].numpy().astype(np.uint8),
                                IMAGE_SIZE[self.name])
                img = Image.fromarray(img, mode=MODE[self.name])
                
                # 应用图像变换
                if self.transform is not None:
                    img = self.transform(img)

                # 应用标签变换
                if self.target_transform is not None:
                    targets[idx] = self.target_transform(targets[idx])

                img_dict[key].append((img, targets[idx]))

        return img_dict

    def process(self):
        """处理原始数据集文件
        
        将原始的 JSON 格式数据文件转换为 PyTorch 张量格式，
        并按照指定比例分割为训练、验证和测试集。
        
        处理流程：
        1. 读取原始 JSON 数据文件
        2. 根据采样比例选择任务子集
        3. 将数据转换为 PyTorch 张量
        4. 按比例分割训练/验证/测试集
        5. 保存处理后的数据到本地
        
        Note:
            - 需要足够的磁盘空间存储处理后的数据
            - 处理过程可能需要较长时间
            - 使用固定随机种子确保可重复性
        """
        # 获取原始数据路径和文件列表
        raw_path = osp.join(self.raw_dir, "all_data")
        files = os.listdir(raw_path)
        files = [f for f in files if f.endswith('.json')]

        # 根据采样比例确定任务数量
        n_tasks = math.ceil(len(files) * self.s_frac)
        random.shuffle(files)
        files = files[:n_tasks]

        print("Preprocess data (Please leave enough space)...")

        idx = 0
        # 遍历每个数据文件进行处理
        for num, file in enumerate(tqdm(files)):

            # 读取 JSON 格式的原始数据
            with open(osp.join(raw_path, file), 'r') as f:
                raw_data = json.load(f)

            # 遍历每个用户的数据，转换为 PyTorch 张量
            for writer, v in raw_data['user_data'].items():
                data, targets = v['x'], v['y']

                # 根据数据维度选择合适的转换方式
                if len(v['x']) > 2:
                    data = torch.tensor(np.stack(data))
                    targets = torch.LongTensor(np.stack(targets))
                else:
                    data = torch.tensor(data)
                    targets = torch.LongTensor(targets)

                # 按比例分割训练集和测试集
                train_data, test_data, train_targets, test_targets =\
                    train_test_split(
                        data,
                        targets,
                        train_size=self.tr_frac,
                        random_state=self.seed
                    )

                # 如果需要验证集，从测试集中再次分割
                if self.val_frac > 0:
                    val_data, test_data, val_targets, test_targets = \
                        train_test_split(
                            test_data,
                            test_targets,
                            train_size=self.val_frac / (1.-self.tr_frac),
                            random_state=self.seed
                        )
                else:
                    val_data, val_targets = None, None
                # 创建任务目录并保存处理后的数据
                save_path = osp.join(self.processed_dir, f"task_{idx}")
                os.makedirs(save_path, exist_ok=True)

                # 保存训练、测试和验证数据到本地文件
                save_local_data(dir_path=save_path,
                                train_data=train_data,
                                train_targets=train_targets,
                                test_data=test_data,
                                test_targets=test_targets,
                                val_data=val_data,
                                val_targets=val_targets)
                idx += 1
