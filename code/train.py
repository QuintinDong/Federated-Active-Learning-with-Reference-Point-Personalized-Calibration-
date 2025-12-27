# -*- coding:utf-8 -*-
"""
FedAvg with DINO Detection and Active Learning
基于FedAvg的联邦学习框架，集成DINO目标检测和主动学习
"""
import argparse
import logging
import os
import random
import shutil
import sys
import time
import copy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import flwr as fl
from flwr.common.logger import log
from flwr.server import ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy.fedavg import FedAvg
from collections import OrderedDict
from logging import DEBUG, INFO
import timeit
from torch.cuda.amp import autocast, GradScaler

from dataloaders.dataset import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, metrics
from val_2D import test_single_volume
from utils.site_encoder import SiteEncoderManager
from utils.dino_detector import DINODetector
from utils.active_learning import ActiveLearningManager
from utils.clustering import FeatureClusterer, DecoderFeatureExtractor


class FedAvgClient(fl.client.Client):
    """FedAvg客户端，集成DINO检测、主动学习和聚类"""
    
    def __init__(self, args, model, trainloader, valloader, site_encoder, dino_detector, 
                 active_learning_manager, clusterer, feature_extractor):
        self.args = args
        self.cid = args.cid
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.site_encoder = site_encoder
        self.dino_detector = dino_detector
        self.active_learning_manager = active_learning_manager
        self.clusterer = clusterer
        self.feature_extractor = feature_extractor
        
        self.current_iter = 0
        self.current_lr = args.base_lr
        self.best_performance = 0.0
        self.amp = (args.amp == 1)
        if self.amp:
            self.scaler = GradScaler()
        
        # 聚类中心历史
        self.cluster_centers_before_upload = None
        self.cluster_centers_after_upload = None
        
        # 初始化主动学习
        if hasattr(trainloader, 'dataset'):
            total_samples = len(trainloader.dataset)
            self.active_learning_manager.initialize(total_samples)
    
    def get_parameters(self, ins):
        """获取模型参数"""
        weights = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        parameters = fl.common.ndarrays_to_parameters(weights)
        return fl.common.GetParametersRes(parameters=parameters, status=fl.common.Status('OK', 'Success'))
    
    def fit(self, ins):
        """训练模型"""
        log(INFO, f'Client {self.cid}: fit round {ins.config.get("iter_global", 0)}')
        
        # 接收服务器参数
        weights = fl.common.parameters_to_ndarrays(ins.parameters)
        config = ins.config
        
        # 保存上传前的聚类中心
        self._save_cluster_centers_before_upload()
        
        # 设置模型参数
        self._set_weights(weights, config)
        
        # 执行训练
        fit_begin = timeit.default_timer()
        loss, metrics_dict = self._train(config)
        
        # 计算上传后的聚类中心
        self._compute_cluster_centers_after_upload()
        
        # 计算聚类中心差值
        center_diff = self._compute_center_difference()
        if center_diff is not None:
            metrics_dict[f'client_{self.cid}_cluster_center_diff'] = center_diff['mean_difference']
        
        # 获取更新后的参数
        weights_prime = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        params_prime = fl.common.ndarrays_to_parameters(weights_prime)
        
        num_examples_train = len(self.trainloader.dataset) if hasattr(self.trainloader, 'dataset') else len(self.trainloader)
        fit_duration = timeit.default_timer() - fit_begin
        metrics_dict['fit_duration'] = fit_duration
        
        return fl.common.FitRes(
            status=fl.common.Status('OK', 'Success'),
            parameters=params_prime,
            num_examples=num_examples_train,
            metrics=metrics_dict
        )
    
    def evaluate(self, ins):
        """评估模型"""
        log(INFO, f'Client {self.cid}: evaluate')
        
        weights = fl.common.parameters_to_ndarrays(ins.parameters)
        config = ins.config
        
        self._set_weights(weights, config)
        loss, metrics_dict = self._validate(config)
        
        return fl.common.EvaluateRes(
            status=fl.common.Status('OK', 'Success'),
            loss=loss,
            num_examples=len(self.valloader),
            metrics=metrics_dict
        )
    
    def _set_weights(self, weights, config):
        """设置模型权重"""
        state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)
        })
        self.model.load_state_dict(state_dict, strict=False)
    
    def _train(self, config):
        """训练函数"""
        self.model.train()
        
        # 优化器
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.current_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2
        )
        
        ce_loss = CrossEntropyLoss(ignore_index=self.args.num_classes)
        
        # 获取站点编码
        site_embedding = self.site_encoder.encode_site(self.cid)
        
        log(INFO, f'Client {self.cid}: {config.get("iters", 20)} iterations per epoch')
        
        all_decoder_features = []
        all_sample_indices = []
        
        for i_iter in range(config.get('iters', 20)):
            # 获取批次数据
            try:
                batch = next(iter(self.trainloader))
            except:
                # 如果数据加载器为空，重新创建迭代器
                train_iter = iter(self.trainloader)
                batch = next(train_iter)
            
            # 处理图像（使用DINO检测）
            images = batch['image']
            labels = batch['label']
            sample_indices = batch.get('idx', torch.arange(len(images)))
            
            # DINO预处理
            processed_images = []
            for img in images:
                img_np = img.numpy()
                if len(img_np.shape) == 2:
                    img_np = np.expand_dims(img_np, axis=0)
                if img_np.shape[0] == 1:
                    img_np = np.repeat(img_np, 3, axis=0)
                img_np = np.transpose(img_np, (1, 2, 0))
                img_np = (img_np * 255).astype(np.uint8)
                
                processed_img, detection_info = self.dino_detector.preprocess_for_segmentation(img_np)
                # 转换回tensor格式
                if len(processed_img.shape) == 3:
                    processed_img = np.transpose(processed_img, (2, 0, 1))
                if processed_img.shape[0] == 3 and self.args.in_chns == 1:
                    processed_img = processed_img[0:1]
                processed_img = torch.from_numpy(processed_img.astype(np.float32) / 255.0)
                processed_images.append(processed_img)
            
            images = torch.stack(processed_images)
            
            if self.args.img_class == 'faz':
                images = images.unsqueeze(1) if len(images.shape) == 3 else images
            elif self.args.img_class == 'odoc' or self.args.img_class == 'polyp':
                if len(images.shape) == 3:
                    images = images.unsqueeze(0)
            
            images = images.cuda()
            labels = labels.cuda()
            
            # 前向传播
            with autocast(enabled=self.amp):
                outputs = self.model(images)
                
                # 提取解码器特征用于聚类
                if isinstance(outputs, tuple):
                    decoder_output = outputs[0]
                    if len(outputs) > 1:
                        # 如果有多个输出，使用第一个作为主要输出
                        main_output = outputs[0]
                    else:
                        main_output = decoder_output
                else:
                    decoder_output = outputs
                    main_output = outputs
                
                # 提取特征用于聚类
                decoder_features = self.feature_extractor.extract_features(decoder_output)
                all_decoder_features.append(decoder_features.detach().cpu())
                all_sample_indices.extend(sample_indices.tolist())
                
                # 计算损失
                loss_ce = ce_loss(main_output, labels.long())
                loss = loss_ce
            
            # 反向传播
            optimizer.zero_grad()
            if self.amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            self.current_iter += 1
            
            # 更新学习率
            lr_ = self.args.base_lr * (1.0 - self.current_iter / self.args.max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            self.current_lr = lr_
            
            if i_iter % 10 == 0:
                log(INFO, f'Client {self.cid}: iteration {self.current_iter}, lr: {self.current_lr:.6f}, loss: {loss.item():.4f}')
        
        # 对解码器输出进行聚类
        if len(all_decoder_features) > 0:
            all_features = torch.cat(all_decoder_features, dim=0).numpy()
            labels, centers = self.clusterer.fit(all_features)
            
            # 选择代表性样本进行主动标注
            if self.active_learning_manager.current_round < self.active_learning_manager.num_rounds:
                representative_indices = self.clusterer.get_representative_samples(
                    all_features, labels, n_samples_per_cluster=1
                )
                # 映射回原始样本索引
                selected_samples = [all_sample_indices[idx] for idx in representative_indices[:self.active_learning_manager.samples_per_round]]
                
                log(INFO, f'Client {self.cid}: Selected {len(selected_samples)} samples for active learning')
                # 注意：这里只是选择样本，实际标注需要人工完成或使用其他方法
        
        # 准备返回的指标
        metrics_dict = {
            f'client_{self.cid}_lr': self.current_lr,
            f'client_{self.cid}_total_loss': loss.item(),
            f'client_{self.cid}_loss_ce': loss.item(),
        }
        
        return loss.item(), metrics_dict
    
    def _validate(self, config):
        """验证函数"""
        self.model.eval()
        val_metrics = self._evaluate(self.args, self.model, self.valloader, self.amp)
        
        if val_metrics['val_mean_dice'] > self.best_performance:
            self.best_performance = val_metrics['val_mean_dice']
            state_dict = self.model.state_dict()
            save_mode_path = os.path.join(
                self.args.snapshot_path, 
                f'client_{self.cid}_best_model.pth'
            )
            torch.save(state_dict, save_mode_path)
            log(INFO, f'Client {self.cid}: save model to {save_mode_path}')
        
        val_metrics = {f'client_{self.cid}_{k}': v for k, v in val_metrics.items()}
        return 0.0, val_metrics
    
    def _evaluate(self, args, model, dataloader, amp=False):
        """评估函数"""
        VAL_METRICS = ['dice', 'hd95', 'recall', 'precision', 'jc', 'specificity', 'ravd']
        metric_list = 0.0
        metrics_dict = {}
        
        for i_batch, sampled_batch in enumerate(dataloader):
            metric_i = test_single_volume(
                sampled_batch['image'], 
                sampled_batch['label'], 
                model, 
                classes=args.num_classes, 
                amp=amp
            )
            metric_list += np.array(metric_i)
        
        metric_list = metric_list / len(dataloader.dataset)
        
        for class_i in range(args.num_classes - 1):
            for metric_i, metric_name in enumerate(VAL_METRICS):
                metrics_dict[f'val_{class_i+1}_{metric_name}'] = metric_list[class_i, metric_i]
        
        for metric_i, metric_name in enumerate(VAL_METRICS):
            metrics_dict[f'val_mean_{metric_name}'] = np.mean(metric_list, axis=0)[metric_i]
        
        return metrics_dict
    
    def _save_cluster_centers_before_upload(self):
        """保存上传前的聚类中心"""
        if self.clusterer.cluster_centers_ is not None:
            self.cluster_centers_before_upload = copy.deepcopy(self.clusterer.cluster_centers_)
    
    def _compute_cluster_centers_after_upload(self):
        """计算上传后的聚类中心"""
        # 基于当前模型在训练数据上的输出重新聚类
        all_features = []
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.trainloader):
                try:
                    images = batch['image']
                    # DINO预处理
                    processed_images = []
                    for img in images:
                        img_np = img.numpy()
                        if len(img_np.shape) == 2:
                            img_np = np.expand_dims(img_np, axis=0)
                        if img_np.shape[0] == 1:
                            img_np = np.repeat(img_np, 3, axis=0)
                        img_np = np.transpose(img_np, (1, 2, 0))
                        img_np = (img_np * 255).astype(np.uint8)
                        
                        processed_img, _ = self.dino_detector.preprocess_for_segmentation(img_np)
                        if len(processed_img.shape) == 3:
                            processed_img = np.transpose(processed_img, (2, 0, 1))
                        if processed_img.shape[0] == 3 and self.args.in_chns == 1:
                            processed_img = processed_img[0:1]
                        processed_img = torch.from_numpy(processed_img.astype(np.float32) / 255.0)
                        processed_images.append(processed_img)
                    
                    images = torch.stack(processed_images)
                    if self.args.img_class == 'faz':
                        images = images.unsqueeze(1) if len(images.shape) == 3 else images
                    elif self.args.img_class == 'odoc' or self.args.img_class == 'polyp':
                        if len(images.shape) == 3:
                            images = images.unsqueeze(0)
                    
                    images = images.cuda()
                    
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        decoder_output = outputs[0]
                    else:
                        decoder_output = outputs
                    
                    features = self.feature_extractor.extract_features(decoder_output)
                    all_features.append(features.cpu().numpy())
                    
                    if batch_idx >= 10:  # 限制批次数量以节省时间
                        break
                except Exception as e:
                    log(INFO, f'Error in computing cluster centers: {e}')
                    break
        
        if len(all_features) > 0:
            all_features = np.vstack(all_features)
            _, centers = self.clusterer.fit(all_features)
            self.cluster_centers_after_upload = centers
    
    def _compute_center_difference(self):
        """计算聚类中心差值"""
        return self.clusterer.compute_center_difference()


def fit_metrics_aggregation_fn(fit_metrics):
    """聚合训练指标"""
    metrics = {k: v for _, client_metrics in fit_metrics for k, v in client_metrics.items()}
    return metrics


def evaluate_metrics_aggregation_fn(evaluate_metrics, args):
    """聚合评估指标"""
    metrics = {k: v for _, client_metrics in evaluate_metrics for k, v in client_metrics.items()}
    
    weights = {}
    for client_id in range(args.min_num_clients):
        first_metric_name = f'client_{client_id}_val_mean_dice'
        for client_num_examples, client_metrics in evaluate_metrics:
            if first_metric_name in client_metrics.keys():
                weights[f'client_{client_id}'] = client_num_examples
    
    def weighted_metric(metric_name):
        num_total_examples = sum(weights.values())
        weighted_metric = [
            weights[f'client_{client_id}'] * metrics[f'client_{client_id}_{metric_name}']
            for client_id in range(args.min_num_clients)
        ]
        return sum(weighted_metric) / num_total_examples if num_total_examples > 0 else 0.0
    
    def mean_metric(metric_name):
        return np.mean([
            metrics[f'client_{client_id}_{metric_name}']
            for client_id in range(args.min_num_clients)
        ])
    
    VAL_METRICS = ['dice', 'hd95', 'recall', 'precision', 'jc', 'specificity', 'ravd']
    for class_i in range(args.num_classes - 1):
        for metric_name in VAL_METRICS:
            metrics[f'val_{class_i+1}_{metric_name}'] = weighted_metric(f'val_{class_i+1}_{metric_name}')
    
    for metric_name in VAL_METRICS:
        metrics[f'val_mean_{metric_name}'] = weighted_metric(f'val_mean_{metric_name}')
        metrics[f'val_avg_mean_{metric_name}'] = mean_metric(f'val_mean_{metric_name}')
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    
    # Flower相关参数
    parser.add_argument('--server_address', type=str, default='[::]:8080', help='gRPC server address')
    parser.add_argument('--gpu', type=int, required=True, help='GPU index')
    parser.add_argument('--role', type=str, required=True, help='Role: server or client')
    parser.add_argument('--iters', type=int, default=20, help='Number of local iterations')
    parser.add_argument('--eval_iters', type=int, default=200, help='Evaluation interval')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='Fraction of clients')
    parser.add_argument('--min_num_clients', type=int, default=2, help='Minimum number of clients')
    
    # 数据相关参数
    parser.add_argument('--root_path', type=str, default='../data/FAZ_h5', help='Data root path')
    parser.add_argument('--exp', type=str, default='fedavg_dino_al', help='Experiment name')
    parser.add_argument('--client', type=str, default='client1', help='Client name')
    parser.add_argument('--sup_type', type=str, default='mask', help='Supervision type')
    parser.add_argument('--model', type=str, default='unet', help='Model name')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--in_chns', type=int, default=1, help='Input channels')
    parser.add_argument('--img_class', type=str, default='faz', help='Image class')
    
    # 训练相关参数
    parser.add_argument('--max_iterations', type=int, default=30000, help='Max iterations')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--base_lr', type=float, default=0.01, help='Base learning rate')
    parser.add_argument('--patch_size', type=list, default=[256, 256], help='Patch size')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('--deterministic', type=int, default=1, help='Deterministic training')
    parser.add_argument('--amp', type=int, default=0, help='Use AMP')
    
    # 客户端ID
    parser.add_argument('--cid', type=int, default=0, help='Client ID')
    
    # 主动学习参数
    parser.add_argument('--al_rounds', type=int, default=5, help='Active learning rounds')
    parser.add_argument('--al_samples_per_round', type=int, default=10, help='Samples per AL round')
    parser.add_argument('--al_strategy', type=str, default='cluster_center', help='AL strategy')
    
    # 聚类参数
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # 设置CUDA
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # 创建保存路径
    snapshot_path = f'../model/{args.exp}'
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    setattr(args, 'snapshot_path', snapshot_path)
    
    # 配置日志
    if args.role == 'server':
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
        shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))
        fl.common.logger.configure('server', filename=os.path.join(snapshot_path, 'server.log'))
        writer = SummaryWriter(snapshot_path + '/log')
    else:
        fl.common.logger.configure(f'client_{args.cid}', filename=os.path.join(snapshot_path, f'client_{args.cid}.log'))
    
    log(INFO, f'Arguments: {args}')
    
    # 加载数据
    db_train = BaseDataSets(
        base_dir=args.root_path,
        split='train',
        transform=transforms.Compose([RandomGenerator(args.patch_size, img_class=args.img_class)]),
        client=args.client,
        sup_type=args.sup_type,
        img_class=args.img_class
    )
    db_val = BaseDataSets(
        base_dir=args.root_path,
        client=args.client,
        split='val',
        img_class=args.img_class
    )
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    valloader = DataLoader(
        db_val,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    # 创建模型
    model = net_factory(args, net_type=args.model, in_chns=args.in_chns, class_num=args.num_classes)
    model.cuda()
    
    # 创建站点编码器
    site_encoder = SiteEncoderManager(num_sites=args.min_num_clients, embedding_dim=128)
    site_encoder.encoder.cuda()
    
    # 创建DINO检测器
    dino_detector = DINODetector(device='cuda')
    
    # 创建主动学习管理器
    active_learning_manager = ActiveLearningManager(
        num_rounds=args.al_rounds,
        samples_per_round=args.al_samples_per_round,
        strategy=args.al_strategy
    )
    
    # 创建聚类器
    clusterer = FeatureClusterer(n_clusters=args.n_clusters, method='kmeans++')
    
    # 创建特征提取器
    feature_extractor = DecoderFeatureExtractor(feature_dim=256)
    # 将投影层移到GPU
    if hasattr(feature_extractor, 'projection'):
        feature_extractor.projection = feature_extractor.projection.cuda()
    
    if args.role == 'server':
        # 服务器端
        def fit_config(server_round):
            return {
                'iter_global': server_round,
                'iters': args.iters,
                'eval_iters': args.eval_iters,
                'batch_size': args.batch_size,
                'stage': 'fit'
            }
        
        def evaluate_config_fn(server_round):
            return {
                'iter_global': server_round,
                'iters': args.iters,
                'eval_iters': args.eval_iters,
                'batch_size': args.batch_size,
                'stage': 'evaluate'
            }
        
        def evaluate_fn(server_round, weights, place):
            model_eval = net_factory(args, net_type=args.model, in_chns=args.in_chns, class_num=args.num_classes)
            state_dict = OrderedDict({
                k: torch.tensor(v) for k, v in zip(model_eval.state_dict().keys(), weights)
            })
            model_eval.load_state_dict(state_dict, strict=True)
            model_eval.cuda()
            model_eval.eval()
            
            VAL_METRICS = ['dice', 'hd95', 'recall', 'precision', 'jc', 'specificity', 'ravd']
            metric_list = 0.0
            metrics_dict = {}
            
            for i_batch, sampled_batch in enumerate(valloader):
                metric_i = test_single_volume(
                    sampled_batch['image'],
                    sampled_batch['label'],
                    model_eval,
                    classes=args.num_classes,
                    amp=(args.amp == 1)
                )
                metric_list += np.array(metric_i)
            
            metric_list = metric_list / len(valloader.dataset)
            
            for class_i in range(args.num_classes - 1):
                for metric_i, metric_name in enumerate(VAL_METRICS):
                    metrics_dict[f'val_{class_i+1}_{metric_name}'] = metric_list[class_i, metric_i]
            
            for metric_i, metric_name in enumerate(VAL_METRICS):
                metrics_dict[f'val_mean_{metric_name}'] = np.mean(metric_list, axis=0)[metric_i]
            
            return 0.0, metrics_dict
        
        strategy = FedAvg(
            fraction_fit=args.sample_fraction,
            min_fit_clients=args.min_num_clients,
            min_available_clients=args.min_num_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=lambda metrics: evaluate_metrics_aggregation_fn(metrics, args),
            accept_failures=False
        )
        
        server = fl.server.Server(
            client_manager=SimpleClientManager(),
            strategy=strategy
        )
        
        fl.server.start_server(
            server_address=args.server_address,
            server=server,
            config=ServerConfig(num_rounds=args.max_iterations, round_timeout=None)
        )
    else:
        # 客户端
        client = FedAvgClient(
            args, model, trainloader, valloader,
            site_encoder, dino_detector,
            active_learning_manager, clusterer, feature_extractor
        )
        fl.client.start_client(server_address=args.server_address, client=client)


if __name__ == '__main__':
    main()

