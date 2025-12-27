# -*- coding:utf-8 -*-
"""
DINO目标检测模块：使用预训练的DINO网络进行目标检测
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2


class DINODetector:
    """DINO目标检测器"""
    
    def __init__(self, model_name='dinov2_vitb14', device='cuda'):
        """
        Args:
            model_name: DINO模型名称
            device: 设备
        """
        self.device = device
        self.model_name = model_name
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """加载预训练的DINO模型"""
        try:
            import torch.hub
            # 加载DINO v2模型
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # 图像预处理
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            print(f"DINO模型 {self.model_name} 加载成功")
        except Exception as e:
            print(f"警告：无法加载DINO模型，将使用简化版本: {e}")
            self.model = None
    
    def extract_features(self, image):
        """
        提取图像特征
        
        Args:
            image: 输入图像，可以是numpy数组或PIL Image
        Returns:
            features: 特征向量
        """
        if self.model is None:
            # 如果DINO模型未加载，返回零特征
            return torch.zeros(768).to(self.device)
        
        # 转换图像格式
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # 灰度图
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 1:  # 单通道
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image.astype(np.uint8))
        
        # 预处理
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(image_tensor)
        
        return features.squeeze(0)
    
    def detect_objects(self, image, threshold=0.5):
        """
        检测图像中的目标区域
        
        Args:
            image: 输入图像
            threshold: 检测阈值
        Returns:
            bboxes: 边界框列表
            features: 特征列表
        """
        if self.model is None:
            # 简化版本：返回整个图像作为检测区域
            h, w = image.shape[:2]
            return [(0, 0, w, h)], [torch.zeros(768).to(self.device)]
        
        # 提取全局特征
        global_features = self.extract_features(image)
        
        # 使用滑动窗口进行局部检测
        h, w = image.shape[:2]
        patch_size = 224
        stride = 112
        
        bboxes = []
        patch_features = []
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[y:y+patch_size, x:x+patch_size]
                patch_feat = self.extract_features(patch)
                
                # 计算与全局特征的相似度
                similarity = torch.cosine_similarity(
                    global_features.unsqueeze(0), 
                    patch_feat.unsqueeze(0)
                ).item()
                
                if similarity > threshold:
                    bboxes.append((x, y, x+patch_size, y+patch_size))
                    patch_features.append(patch_feat)
        
        if len(bboxes) == 0:
            # 如果没有检测到目标，返回整个图像
            bboxes = [(0, 0, w, h)]
            patch_features = [global_features]
        
        return bboxes, patch_features
    
    def preprocess_for_segmentation(self, image):
        """
        预处理图像用于分割任务
        
        Args:
            image: 输入图像
        Returns:
            processed_image: 处理后的图像
            detection_info: 检测信息（边界框、特征等）
        """
        bboxes, features = self.detect_objects(image)
        
        # 将检测信息编码到图像中（可选）
        # 这里简单返回原图像和检测信息
        detection_info = {
            'bboxes': bboxes,
            'features': features,
            'num_detections': len(bboxes)
        }
        
        return image, detection_info

