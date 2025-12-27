# -*- coding:utf-8 -*-
"""
测试FedAvg with DINO and Active Learning模型
"""
import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import h5py
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets
from networks.net_factory import net_factory
from val_2D import test_single_volume
from utils.site_encoder import SiteEncoderManager
from utils.dino_detector import DINODetector


def get_client_ids(client, base_dir):
    """获取客户端数据ID"""
    client1_test_set = 'Domain1/test/' + pd.Series(os.listdir(base_dir + "/Domain1/test"))
    client1_training_set = 'Domain1/train/' + pd.Series(os.listdir(base_dir + "/Domain1/train"))
    client2_test_set = 'Domain2/test/' + pd.Series(os.listdir(base_dir + "/Domain2/test"))
    client2_training_set = 'Domain2/train/' + pd.Series(os.listdir(base_dir + "/Domain2/train"))
    client3_test_set = 'Domain3/test/' + pd.Series(os.listdir(base_dir + "/Domain3/test"))
    client3_training_set = 'Domain3/train/' + pd.Series(os.listdir(base_dir + "/Domain3/train"))
    client4_test_set = 'Domain4/test/' + pd.Series(os.listdir(base_dir + "/Domain4/test"))
    client4_training_set = 'Domain4/train/' + pd.Series(os.listdir(base_dir + "/Domain4/train"))
    client5_test_set = 'Domain5/test/' + pd.Series(os.listdir(base_dir + "/Domain5/test"))
    client5_training_set = 'Domain5/train/' + pd.Series(os.listdir(base_dir + "/Domain5/train"))
    
    client1_test_set = client1_test_set.tolist()
    client1_training_set = client1_training_set.tolist()
    client2_test_set = client2_test_set.tolist()
    client2_training_set = client2_training_set.tolist()
    client3_test_set = client3_test_set.tolist()
    client3_training_set = client3_training_set.tolist()
    client4_test_set = client4_test_set.tolist()
    client4_training_set = client4_training_set.tolist()
    client5_test_set = client5_test_set.tolist()
    client5_training_set = client5_training_set.tolist()
    
    if client == "client1":
        return [client1_training_set, client1_test_set]
    elif client == "client2":
        return [client2_training_set, client2_test_set]
    elif client == "client3":
        return [client3_training_set, client3_test_set]
    elif client == "client4":
        return [client4_training_set, client4_test_set]
    elif client == "client5":
        return [client5_training_set, client5_test_set]
    elif client == "client_all":
        client_train_all = client1_training_set + client2_training_set + client3_training_set + \
                          client4_training_set + client5_training_set
        client_test_all = client1_test_set + client2_test_set + client3_test_set + \
                         client4_test_set + client5_test_set
        return [client_train_all, client_test_all]
    else:
        return "ERROR KEY"


def get_client_ids_polyp(client, base_dir):
    """获取polyp数据集的客户端ID"""
    client1_test_set = 'Domain1/test/' + pd.Series(os.listdir(base_dir + "/Domain1/test"))
    client1_training_set = 'Domain1/train/' + pd.Series(os.listdir(base_dir + "/Domain1/train"))
    client2_test_set = 'Domain2/test/' + pd.Series(os.listdir(base_dir + "/Domain2/test"))
    client2_training_set = 'Domain2/train/' + pd.Series(os.listdir(base_dir + "/Domain2/train"))
    client3_test_set = 'Domain3/test/' + pd.Series(os.listdir(base_dir + "/Domain3/test"))
    client3_training_set = 'Domain3/train/' + pd.Series(os.listdir(base_dir + "/Domain3/train"))
    client4_test_set = 'Domain4/test/' + pd.Series(os.listdir(base_dir + "/Domain4/test"))
    client4_training_set = 'Domain4/train/' + pd.Series(os.listdir(base_dir + "/Domain4/train"))
    
    client1_test_set = client1_test_set.tolist()
    client1_training_set = client1_training_set.tolist()
    client2_test_set = client2_test_set.tolist()
    client2_training_set = client2_training_set.tolist()
    client3_test_set = client3_test_set.tolist()
    client3_training_set = client3_training_set.tolist()
    client4_test_set = client4_test_set.tolist()
    client4_training_set = client4_training_set.tolist()
    
    if client == "client1":
        return [client1_training_set, client1_test_set]
    elif client == "client2":
        return [client2_training_set, client2_test_set]
    elif client == "client3":
        return [client3_training_set, client3_test_set]
    elif client == "client4":
        return [client4_training_set, client4_test_set]
    elif client == "client_all":
        client_train_all = client1_training_set + client2_training_set + client3_training_set + \
                          client4_training_set
        client_test_all = client1_test_set + client2_test_set + client3_test_set + \
                         client4_test_set
        return [client_train_all, client_test_all]
    else:
        return "ERROR KEY"


def test_single_image(case, net, test_save_path, FLAGS, dino_detector):
    """测试单个图像"""
    h5f = h5py.File(FLAGS.root_path + "/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['mask'][:]
    prediction = np.zeros_like(label)
    
    # DINO预处理
    if len(image.shape) == 2:
        img_np = np.expand_dims(image, axis=0)
    else:
        img_np = image
    if img_np.shape[0] == 1:
        img_np = np.repeat(img_np, 3, axis=0)
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = (img_np * 255).astype(np.uint8)
    
    processed_img, detection_info = dino_detector.preprocess_for_segmentation(img_np)
    
    if len(image.shape) == 3:
        slice = processed_img
        if len(slice.shape) == 3:
            slice = np.transpose(slice, (2, 0, 1))
        if slice.shape[0] == 3 and FLAGS.in_chns == 1:
            slice = slice[0:1]
        input = torch.from_numpy(slice).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction = out
            item = case.split("/")[-1].split(".")[0]
            import cv2
            cv2.imwrite(test_save_path + '/pre/' + item + "_pred.png", prediction * 85.)
            cv2.imwrite(test_save_path + '/pre/' + item + "_gt.png", label * 85.)
    
    elif len(image.shape) == 2:
        slice = processed_img
        if len(slice.shape) == 3:
            slice = np.transpose(slice, (2, 0, 1))
        if slice.shape[0] == 3 and FLAGS.in_chns == 1:
            slice = slice[0:1]
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input)[0], dim=1), dim=1).squeeze(1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction = out
            item = case.split("/")[-1].split(".")[0]
            import cv2
            cv2.imwrite(test_save_path + '/pre/' + item + "_pred.png", prediction * 127.)
            cv2.imwrite(test_save_path + '/pre/' + item + "_gt.png", label * 127.)
    
    from medpy import metric
    
    def calculate_metric_percase(pred, gt):
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        if pred.sum() > 0:
            dice = metric.binary.dc(pred, gt)
            jaccard = metric.binary.jc(pred, gt)
            hd95 = metric.binary.hd95(pred, gt)
            assd = metric.binary.assd(pred, gt)
            se = metric.binary.sensitivity(pred, gt)
            sp = metric.binary.specificity(pred, gt)
            recall = metric.binary.recall(pred, gt)
            precision = metric.binary.precision(pred, gt)
            return dice, jaccard, hd95, assd, se, sp, recall, precision
        else:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    if FLAGS.img_class == 'faz' or FLAGS.img_class == 'polyp':
        if np.sum(prediction) == 0:
            print("pred_size", prediction.shape)
            prediction = cv2.circle(np.array(prediction).astype(np.uint8), (192, 192), 1, (1, 1, 1), -1)
        metric = calculate_metric_percase(prediction == 1, label == 1)
        return metric
    
    if FLAGS.img_class == 'odoc':
        if np.sum(prediction) == 0:
            prediction = cv2.circle(np.array(prediction).astype(np.uint8), (192, 192), 1, (1, 1, 1), -1)
        metric1 = calculate_metric_percase(prediction == 1, label == 1)
        metric2 = calculate_metric_percase(prediction >= 1, label >= 1)
        return metric1, metric2


def Inference(FLAGS):
    """推理函数"""
    if FLAGS.img_class == 'odoc' or FLAGS.img_class == 'faz':
        train_ids, test_ids = get_client_ids(FLAGS.client, FLAGS.root_path)
    elif FLAGS.img_class == 'polyp':
        train_ids, test_ids = get_client_ids_polyp(FLAGS.client, FLAGS.root_path)
    
    image_list = test_ids
    snapshot_path = "../model/{}/".format(FLAGS.exp)
    test_save_path = "../model/{}_test/{}/".format(FLAGS.exp, FLAGS.client)
    
    import shutil
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path + '/pre/')
    
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    net = net_factory(FLAGS, net_type=FLAGS.model, in_chns=FLAGS.in_chns, class_num=FLAGS.num_classes)
    
    save_mode_path = os.path.join(
        snapshot_path,
        '{}_{}_best_model.pth'.format(FLAGS.client, FLAGS.model).replace("client", "client_")
    )
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    
    # 创建DINO检测器
    dino_detector = DINODetector(device='cuda')
    
    names = []
    dices = []
    jaccards = []
    HD95s = []
    ASSDs = []
    SEs = []
    SPs = []
    Recs = []
    Pres = []
    
    if FLAGS.img_class == 'faz' or FLAGS.img_class == 'polyp':
        for case in tqdm(image_list):
            print(case)
            metric = test_single_image(case, net, test_save_path, FLAGS, dino_detector)
            dice, jaccard, HD95, ASSD, SE, SP, Rec, Pre = metric[0], metric[1], metric[2], metric[3], metric[4], metric[5], metric[6], metric[7]
            
            names.append(str(case))
            dices.append(dice)
            jaccards.append(jaccard)
            HD95s.append(HD95)
            ASSDs.append(ASSD)
            SEs.append(SE)
            SPs.append(SP)
            Recs.append(Rec)
            Pres.append(Pre)
        
        dataframe = pd.DataFrame({
            'name': names, 'dice': dices, 'jaccard': jaccards, 'HD95': HD95s,
            'ASSD': ASSDs, 'SE': SEs, 'SP': SPs, 'Rec': Recs, 'Pre': Pres
        })
        dataframe.to_csv(test_save_path + "result.csv", index=False, sep=',')
        print('Counting CSV generated!')
        
        mean_std_resultframe = pd.DataFrame({
            'name': ['mean', 'std'],
            'dice': [np.mean(dices), np.std(dices)],
            'jaccard': [np.mean(jaccards), np.std(jaccards)],
            'HD95': [np.mean(HD95s), np.std(HD95s)],
            'ASSD': [np.mean(ASSDs), np.std(ASSDs)],
            'SE': [np.mean(SEs), np.std(SEs)],
            'SP': [np.mean(SPs), np.std(SPs)],
            'Rec': [np.mean(Recs), np.std(Recs)],
            'Pre': [np.mean(Pres), np.std(Pres)]
        })
        mean_std_resultframe.to_csv(test_save_path + "mean_std_result.csv", index=False, sep=',')
        print('Mean and Std CSV generated!')
        avg_dice = np.mean(dices)
    
    if FLAGS.img_class == 'odoc':
        dices1 = []
        jaccards1 = []
        HD95s1 = []
        ASSDs1 = []
        SEs1 = []
        SPs1 = []
        Recs1 = []
        Pres1 = []
        dices2 = []
        jaccards2 = []
        HD95s2 = []
        ASSDs2 = []
        SEs2 = []
        SPs2 = []
        Recs2 = []
        Pres2 = []
        
        for case in tqdm(image_list):
            print(case)
            metric1, metric2 = test_single_image(case, net, test_save_path, FLAGS, dino_detector)
            dice, jaccard, HD95, ASSD, SE, SP, Rec, Pre = metric1[0], metric1[1], metric1[2], metric1[3], metric1[4], metric1[5], metric1[6], metric1[7]
            names.append(str(case))
            dices1.append(dice)
            jaccards1.append(jaccard)
            HD95s1.append(HD95)
            ASSDs1.append(ASSD)
            SEs1.append(SE)
            SPs1.append(SP)
            Recs1.append(Rec)
            Pres1.append(Pre)
            
            dice, jaccard, HD95, ASSD, SE, SP, Rec, Pre = metric2[0], metric2[1], metric2[2], metric2[3], metric2[4], metric2[5], metric2[6], metric2[7]
            dices2.append(dice)
            jaccards2.append(jaccard)
            HD95s2.append(HD95)
            ASSDs2.append(ASSD)
            SEs2.append(SE)
            SPs2.append(SP)
            Recs2.append(Rec)
            Pres2.append(Pre)
        
        dataframe = pd.DataFrame({
            'name': names,
            'dice_cup': dices1, 'jaccard_cup': jaccards1, 'HD95_cup': HD95s1,
            'ASSD_cup': ASSDs1, 'SE_cup': SEs1, 'SP_cup': SPs1, 'Rec_cup': Recs1, 'Pre_cup': Pres1,
            'dice_disc': dices2, 'jaccard_disc': jaccards2, 'HD95_disc': HD95s2,
            'ASSD_disc': ASSDs2, 'SE_disc': SEs2, 'SP_disc': SPs2, 'Rec_disc': Recs2, 'Pre_disc': Pres2
        })
        dataframe.to_csv(test_save_path + "result.csv", index=False, sep=',')
        print('Counting CSV generated!')
        
        mean_std_resultframe = pd.DataFrame({
            'name': ['mean', 'std'],
            'dice_cup': [np.mean(dices1), np.std(dices1)],
            'jaccard_cup': [np.mean(jaccards1), np.std(jaccards1)],
            'HD95_cup': [np.mean(HD95s1), np.std(HD95s1)],
            'ASSD_cup': [np.mean(ASSDs1), np.std(ASSDs1)],
            'SE_cup': [np.mean(SEs1), np.std(SEs1)],
            'SP_cup': [np.mean(SPs1), np.std(SPs1)],
            'Rec_cup': [np.mean(Recs1), np.std(Recs1)],
            'Pre_cup': [np.mean(Pres1), np.std(Pres1)],
            'dice_disc': [np.mean(dices2), np.std(dices2)],
            'jaccard_disc': [np.mean(jaccards2), np.std(jaccards2)],
            'HD95_disc': [np.mean(HD95s2), np.std(HD95s2)],
            'ASSD_disc': [np.mean(ASSDs2), np.std(ASSDs2)],
            'SE_disc': [np.mean(SEs2), np.std(SEs2)],
            'SP_disc': [np.mean(SPs2), np.std(SPs2)],
            'Rec_disc': [np.mean(Recs2), np.std(Recs2)],
            'Pre_disc': [np.mean(Pres2), np.std(Pres2)]
        })
        mean_std_resultframe.to_csv(test_save_path + "mean_std_result.csv", index=False, sep=',')
        print('Mean and Std CSV generated!')
        avg_dice = np.mean(dices1)
    
    return avg_dice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='../data/ODOC_h5', help='Data root path')
    parser.add_argument('--exp', type=str, default='fedavg_dino_al', help='Experiment name')
    parser.add_argument('--model', type=str, default='unet', help='Model name')
    parser.add_argument('--client', type=str, default='client5', help='Client')
    parser.add_argument('--data_type', type=str, default='octa', help='Data type')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--in_chns', type=int, default=3, help='Input channels')
    parser.add_argument('--sup_type', type=str, default="scribble", help='Supervision type')
    parser.add_argument('--snapshot_path', type=str, default="../model/fedavg_dino_al", help='Snapshot path')
    parser.add_argument('--img_class', type=str, default='faz', help='Image class')
    parser.add_argument('--min_num_clients', type=int, default=5, help='Min num clients')
    parser.add_argument('--cid', type=int, default=0, help='Client ID')
    
    FLAGS = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    seed = 2022
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    mean_dice = Inference(FLAGS)
    print(f'Mean Dice: {mean_dice}')

