# -*- coding:utf-8 -*-
"""
运行FedAvg with DINO and Active Learning的脚本
"""
from multiprocessing import Pool
import os
import time
import argparse


def run_cmd(cmd_str, debug):
    print('Running "{}"\n'.format(cmd_str))
    if debug == 0:
        os.system(cmd_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True, help='Communication port')
    parser.add_argument('--debug', type=int, default=0, help='Debug mode')
    parser.add_argument('--exp', type=str, required=True, help='Experiment name')
    parser.add_argument('--gpus', nargs='+', type=int, required=True, help='GPU indexes')
    parser.add_argument('--base_lr', type=float, default=0.01, help='Base learning rate')
    parser.add_argument('--model', type=str, default='unet', help='Model name')
    parser.add_argument('--img_class', type=str, default='faz', help='Image class')
    parser.add_argument('--max_iterations', type=int, default=30000, help='Max iterations')
    parser.add_argument('--iters', type=int, default=20, help='Local iterations')
    parser.add_argument('--eval_iters', type=int, default=200, help='Evaluation interval')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--amp', type=int, default=0, help='Use AMP')
    parser.add_argument('--al_rounds', type=int, default=5, help='Active learning rounds')
    parser.add_argument('--al_samples_per_round', type=int, default=10, help='Samples per AL round')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters')
    
    args = parser.parse_args()
    
    assert args.img_class in ['odoc', 'faz', 'polyp']
    assert len(args.gpus) >= 6  # 至少需要6个GPU（1个服务器+5个客户端）
    
    # 根据图像类型设置参数
    if args.img_class == 'faz':
        root_path = '../data/FAZ_h5'
        num_classes = 2
        in_chns = 1
        mask_dict = {
            'client1': 'scribble_noisy',
            'client2': 'keypoint',
            'client3': 'block',
            'client4': 'box',
            'client5': 'scribble'
        }
    elif args.img_class == 'odoc':
        root_path = '../data/ODOC_h5'
        num_classes = 3
        in_chns = 3
        mask_dict = {
            'client1': 'scribble',
            'client2': 'scribble_noisy',
            'client3': 'scribble_noisy',
            'client4': 'keypoint',
            'client5': 'block'
        }
    elif args.img_class == 'polyp':
        root_path = '../data/Polypdata_h5'
        num_classes = 2
        in_chns = 3
        mask_dict = {
            'client1': 'keypoint',
            'client2': 'scribble',
            'client3': 'box',
            'client4': 'block'
        }
    
    # 构建通用命令
    common_cmd = (
        f'python flower_fedavg_dino_al.py '
        f'--root_path {root_path} '
        f'--num_classes {num_classes} '
        f'--in_chns {in_chns} '
        f'--img_class {args.img_class} '
        f'--exp {args.exp} '
        f'--model {args.model} '
        f'--max_iterations {args.max_iterations} '
        f'--iters {args.iters} '
        f'--eval_iters {args.eval_iters} '
        f'--batch_size {args.batch_size} '
        f'--base_lr {args.base_lr} '
        f'--amp {args.amp} '
        f'--server_address [::]:{args.port} '
        f'--min_num_clients {len(mask_dict)} '
        f'--al_rounds {args.al_rounds} '
        f'--al_samples_per_round {args.al_samples_per_round} '
        f'--n_clusters {args.n_clusters}'
    )
    
    # 构建客户端命令
    client_args = (
        [f'--role server --client client_all --sup_type mask --gpu {args.gpus[0]}'] +
        [f'--role client --cid {i} --client {client} --sup_type {sup_type} --gpu {args.gpus[i + 1]}'
         for i, (client, sup_type) in enumerate(mask_dict.items())]
    )
    
    pool = Pool(len(client_args))
    for i in range(len(client_args)):
        pool.apply_async(run_cmd, [f'{common_cmd} {client_args[i]}', args.debug])
        if args.debug == 0:
            if i == 0:
                time.sleep(10)  # 服务器先启动
            else:
                time.sleep(7)  # 客户端依次启动
        else:
            time.sleep(1)
    
    pool.close()
    pool.join()

