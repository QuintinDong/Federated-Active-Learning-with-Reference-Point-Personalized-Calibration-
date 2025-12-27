# -*- coding:utf-8 -*-
"""
聚类算法模块：对解码器输出进行聚类，选取代表性样本
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import copy


class FeatureClusterer:
    """特征聚类器，用于对解码器输出进行聚类"""
    
    def __init__(self, n_clusters=10, method='kmeans'):
        """
        Args:
            n_clusters: 聚类数量
            method: 聚类方法 ('kmeans', 'kmeans++')
        """
        self.n_clusters = n_clusters
        self.method = method
        self.cluster_centers_ = None
        self.labels_ = None
        self.previous_centers_ = None
    
    def fit(self, features):
        """
        对特征进行聚类
        
        Args:
            features: 特征矩阵，形状为 [n_samples, feature_dim]
        Returns:
            labels: 聚类标签
            centers: 聚类中心
        """
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        # 保存之前的聚类中心
        self.previous_centers_ = copy.deepcopy(self.cluster_centers_)
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=self.n_clusters, 
                        init='k-means++' if self.method == 'kmeans++' else 'random',
                        n_init=10, 
                        random_state=42)
        self.labels_ = kmeans.fit_predict(features)
        self.cluster_centers_ = kmeans.cluster_centers_
        
        return self.labels_, self.cluster_centers_
    
    def get_representative_samples(self, features, labels=None, n_samples_per_cluster=1):
        """
        从每个聚类中选择代表性样本
        
        Args:
            features: 特征矩阵
            labels: 聚类标签（如果为None，则使用之前fit的结果）
            n_samples_per_cluster: 每个聚类选择的样本数
        Returns:
            representative_indices: 代表性样本的索引
        """
        if labels is None:
            if self.labels_ is None:
                raise ValueError("需要先调用fit方法或提供labels")
            labels = self.labels_
        
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        representative_indices = []
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_features = features[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_features) == 0:
                continue
            
            # 选择距离聚类中心最近的样本作为代表性样本
            center = self.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_features - center, axis=1)
            
            # 选择最近的n_samples_per_cluster个样本
            n_select = min(n_samples_per_cluster, len(cluster_indices))
            top_indices = np.argsort(distances)[:n_select]
            representative_indices.extend(cluster_indices[top_indices].tolist())
        
        return representative_indices
    
    def compute_center_difference(self):
        """
        计算当前聚类中心与之前聚类中心的差值
        
        Returns:
            difference: 中心差值（如果之前没有中心，返回None）
        """
        if self.previous_centers_ is None or self.cluster_centers_ is None:
            return None
        
        # 计算每个聚类中心的移动距离
        differences = np.linalg.norm(
            self.cluster_centers_ - self.previous_centers_, 
            axis=1
        )
        
        return {
            'mean_difference': np.mean(differences),
            'max_difference': np.max(differences),
            'min_difference': np.min(differences),
            'std_difference': np.std(differences),
            'per_cluster_differences': differences
        }
    
    def update_clusters(self, features, adaptive_n_clusters=False):
        """
        更新聚类（用于在线学习）
        
        Args:
            features: 新特征
            adaptive_n_clusters: 是否自适应调整聚类数量
        """
        if adaptive_n_clusters:
            # 可以根据数据分布自适应调整聚类数量
            # 这里简化处理，保持固定数量
            pass
        
        return self.fit(features)


class DecoderFeatureExtractor:
    """从解码器输出中提取特征用于聚类"""
    
    def __init__(self, feature_dim=256):
        """
        Args:
            feature_dim: 特征维度
        """
        self.feature_dim = feature_dim
    
    def extract_features(self, decoder_output):
        """
        从解码器输出中提取特征
        
        Args:
            decoder_output: 解码器输出，可以是特征图或最终输出
        Returns:
            features: 提取的特征向量
        """
        if isinstance(decoder_output, tuple):
            # 如果是元组，取第一个元素（通常是主要输出）
            decoder_output = decoder_output[0]
        
        # 如果是特征图，进行全局平均池化
        if len(decoder_output.shape) == 4:  # [B, C, H, W]
            # 全局平均池化
            features = torch.mean(decoder_output.view(
                decoder_output.size(0), 
                decoder_output.size(1), 
                -1
            ), dim=2)  # [B, C]
        elif len(decoder_output.shape) == 3:  # [B, H, W]
            features = decoder_output.view(decoder_output.size(0), -1)  # [B, H*W]
        else:
            features = decoder_output
        
        # 如果特征维度不匹配，进行投影
        if features.size(-1) != self.feature_dim:
            if not hasattr(self, 'projection'):
                self.projection = nn.Linear(features.size(-1), self.feature_dim)
                if features.is_cuda:
                    self.projection = self.projection.cuda()
            features = self.projection(features)
        
        return features

