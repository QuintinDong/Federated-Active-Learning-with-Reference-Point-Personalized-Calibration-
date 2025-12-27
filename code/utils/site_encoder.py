# -*- coding:utf-8 -*-
"""
站点编码模块：为每个站点生成唯一编码
"""
import torch
import torch.nn as nn
import numpy as np


class SiteEncoder(nn.Module):
    """站点编码器，为每个站点生成唯一的嵌入向量"""
    
    def __init__(self, num_sites, embedding_dim=128):
        """
        Args:
            num_sites: 站点数量
            embedding_dim: 嵌入向量维度
        """
        super(SiteEncoder, self).__init__()
        self.num_sites = num_sites
        self.embedding_dim = embedding_dim
        
        # 使用可学习的嵌入层
        self.embedding = nn.Embedding(num_sites, embedding_dim)
        self._init_embeddings()
    
    def _init_embeddings(self):
        """初始化嵌入向量"""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
    
    def forward(self, site_ids):
        """
        Args:
            site_ids: 站点ID，形状为 [batch_size] 或标量
        Returns:
            site_embeddings: 站点嵌入向量，形状为 [batch_size, embedding_dim] 或 [embedding_dim]
        """
        if isinstance(site_ids, int):
            site_ids = torch.tensor([site_ids], device=self.embedding.weight.device)
        elif not isinstance(site_ids, torch.Tensor):
            site_ids = torch.tensor(site_ids, device=self.embedding.weight.device)
        
        return self.embedding(site_ids)
    
    def get_site_embedding(self, site_id):
        """获取单个站点的嵌入向量"""
        return self.forward(site_id)
    
    def get_all_embeddings(self):
        """获取所有站点的嵌入向量"""
        site_ids = torch.arange(self.num_sites, device=self.embedding.weight.device)
        return self.forward(site_ids)


class SiteEncoderManager:
    """站点编码管理器"""
    
    def __init__(self, num_sites, embedding_dim=128):
        self.num_sites = num_sites
        self.embedding_dim = embedding_dim
        self.encoder = SiteEncoder(num_sites, embedding_dim)
        self.site_features = {}  # 存储每个站点的特征统计信息
    
    def encode_site(self, site_id, features=None):
        """
        编码站点，可以结合站点特征
        
        Args:
            site_id: 站点ID
            features: 可选的站点特征（如数据分布统计等）
        """
        embedding = self.encoder.get_site_embedding(site_id)
        
        if features is not None:
            # 如果有额外特征，可以融合
            if isinstance(features, torch.Tensor):
                # 简单的特征融合
                if features.dim() == 1:
                    features = features.unsqueeze(0)
                # 将特征投影到嵌入空间
                feature_proj = nn.Linear(features.size(-1), self.embedding_dim).to(embedding.device)
                feature_embedding = feature_proj(features)
                embedding = embedding + 0.1 * feature_embedding
        
        return embedding
    
    def update_site_features(self, site_id, features):
        """更新站点特征统计"""
        self.site_features[site_id] = features
    
    def get_site_code(self, site_id):
        """获取站点的编码（用于标识）"""
        return f"site_{site_id:03d}"

