# -*- coding:utf-8 -*-
"""
主动学习模块：管理主动学习流程，包括样本选择、标注管理等
"""
import torch
import numpy as np
import os
import json
from typing import List, Dict, Tuple
from collections import defaultdict


class ActiveLearningManager:
    """主动学习管理器"""
    
    def __init__(self, num_rounds=5, samples_per_round=10, strategy='uncertainty'):
        """
        Args:
            num_rounds: 主动学习轮数
            samples_per_round: 每轮选择的样本数
            strategy: 样本选择策略 ('uncertainty', 'diversity', 'cluster_center')
        """
        self.num_rounds = num_rounds
        self.samples_per_round = samples_per_round
        self.strategy = strategy
        self.current_round = 0
        self.labeled_samples = set()  # 已标注样本索引
        self.unlabeled_samples = set()  # 未标注样本索引
        self.sample_uncertainties = {}  # 样本不确定性分数
        self.sample_features = {}  # 样本特征
        self.annotation_history = []  # 标注历史
    
    def initialize(self, total_samples):
        """初始化，设置所有样本为未标注"""
        self.unlabeled_samples = set(range(total_samples))
        self.labeled_samples = set()
        self.current_round = 0
    
    def select_samples_for_annotation(self, model, dataloader, clusterer=None):
        """
        选择需要标注的样本
        
        Args:
            model: 当前模型
            dataloader: 数据加载器
            clusterer: 聚类器（可选）
        Returns:
            selected_indices: 选中的样本索引列表
        """
        if self.current_round >= self.num_rounds:
            return []
        
        if self.strategy == 'uncertainty':
            return self._select_by_uncertainty(model, dataloader)
        elif self.strategy == 'diversity':
            return self._select_by_diversity(model, dataloader)
        elif self.strategy == 'cluster_center':
            return self._select_by_cluster_center(model, dataloader, clusterer)
        else:
            return self._select_by_uncertainty(model, dataloader)
    
    def _select_by_uncertainty(self, model, dataloader):
        """基于不确定性选择样本"""
        uncertainties = []
        indices = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                image = batch['image'].cuda()
                sample_idx = batch.get('idx', batch_idx)
                
                # 计算预测和不确定性
                output = model(image)
                if isinstance(output, tuple):
                    output = output[0]
                
                probs = torch.softmax(output, dim=1)
                # 使用熵作为不确定性度量
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                uncertainty = entropy.mean().item()
                
                # 只考虑未标注的样本
                if isinstance(sample_idx, torch.Tensor):
                    sample_idx = sample_idx.item()
                if sample_idx not in self.labeled_samples:
                    uncertainties.append(uncertainty)
                    indices.append(sample_idx)
                    self.sample_uncertainties[sample_idx] = uncertainty
        
        # 选择不确定性最高的样本
        if len(uncertainties) == 0:
            return []
        
        sorted_indices = sorted(zip(uncertainties, indices), reverse=True)
        selected = [idx for _, idx in sorted_indices[:self.samples_per_round]]
        
        return selected
    
    def _select_by_diversity(self, model, dataloader):
        """基于多样性选择样本"""
        # 简化实现：结合不确定性和特征多样性
        selected = self._select_by_uncertainty(model, dataloader)
        return selected
    
    def _select_by_cluster_center(self, model, dataloader, clusterer):
        """基于聚类中心选择代表性样本"""
        if clusterer is None:
            return self._select_by_uncertainty(model, dataloader)
        
        # 提取所有未标注样本的特征
        features_list = []
        indices_list = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                image = batch['image'].cuda()
                sample_idx = batch.get('idx', batch_idx)
                
                if isinstance(sample_idx, torch.Tensor):
                    sample_idx = sample_idx.item()
                
                if sample_idx not in self.labeled_samples:
                    output = model(image)
                    # 提取特征（这里简化处理，使用解码器输出）
                    if isinstance(output, tuple):
                        feat = output[0]  # 使用第一个输出作为特征
                    else:
                        feat = output
                    
                    # 全局平均池化
                    if len(feat.shape) == 4:
                        feat = torch.mean(feat.view(feat.size(0), feat.size(1), -1), dim=2)
                    elif len(feat.shape) == 3:
                        feat = feat.view(feat.size(0), -1)
                    
                    features_list.append(feat.cpu().numpy())
                    indices_list.append(sample_idx)
        
        if len(features_list) == 0:
            return []
        
        # 合并特征
        all_features = np.vstack(features_list)
        
        # 进行聚类
        labels, centers = clusterer.fit(all_features)
        
        # 选择代表性样本
        representative_indices = clusterer.get_representative_samples(
            all_features, labels, n_samples_per_cluster=1
        )
        
        # 映射回原始索引
        selected = [indices_list[idx] for idx in representative_indices[:self.samples_per_round]]
        
        return selected
    
    def add_annotations(self, sample_indices, annotations):
        """
        添加标注
        
        Args:
            sample_indices: 样本索引列表
            annotations: 标注数据（可以是标签或标注文件路径）
        """
        for idx, annotation in zip(sample_indices, annotations):
            self.labeled_samples.add(idx)
            if idx in self.unlabeled_samples:
                self.unlabeled_samples.remove(idx)
            
            self.annotation_history.append({
                'round': self.current_round,
                'sample_idx': idx,
                'annotation': annotation
            })
        
        self.current_round += 1
    
    def get_labeled_mask(self, total_samples):
        """获取已标注样本的掩码"""
        mask = np.zeros(total_samples, dtype=bool)
        for idx in self.labeled_samples:
            if idx < total_samples:
                mask[idx] = True
        return mask
    
    def save_state(self, save_path):
        """保存主动学习状态"""
        state = {
            'current_round': self.current_round,
            'labeled_samples': list(self.labeled_samples),
            'unlabeled_samples': list(self.unlabeled_samples),
            'annotation_history': self.annotation_history,
            'num_rounds': self.num_rounds,
            'samples_per_round': self.samples_per_round,
            'strategy': self.strategy
        }
        
        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, load_path):
        """加载主动学习状态"""
        with open(load_path, 'r') as f:
            state = json.load(f)
        
        self.current_round = state['current_round']
        self.labeled_samples = set(state['labeled_samples'])
        self.unlabeled_samples = set(state['unlabeled_samples'])
        self.annotation_history = state['annotation_history']
        self.num_rounds = state['num_rounds']
        self.samples_per_round = state['samples_per_round']
        self.strategy = state['strategy']

