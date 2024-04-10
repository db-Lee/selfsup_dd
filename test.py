import random
import argparse
import numpy as np

import torch

from data.wrapper import get_loader
from data.augmentation import NUM_CLASSES, ParamDiffAug
from algorithms.wrapper import get_algorithm
    
def main(args):
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    # default augment
    args.dsa_param = ParamDiffAug()
    args.dsa_strategy = 'color_crop_cutout_flip_scale_rotate'

    # seed
    if args.seed is None:
        args.seed = random.randint(0, 9999)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # data    
    x_syn = torch.load(f"{args.synthetic_data_dir}/{args.source_data_name}/{args.method}/x_syn.pt", map_location="cpu").detach()
    y_syn = torch.load(f"{args.synthetic_data_dir}/{args.source_data_name}/{args.method}/y_syn.pt", map_location="cpu").detach()
    if args.method == "kip" or args.method == "frepo" or args.method == "krr_st":
        y_syn = y_syn.float()
    else:
        y_syn = y_syn.long()
    if args.method == "krr_st" :
        args.num_pretrain_classes = y_syn.shape[-1]
    else:
        args.num_pretrain_classes = NUM_CLASSES[args.source_data_name]

    print(args)

    # algo
    if args.method == "random" or args.method == "kmeans" or args.method == "dsa" or args.method == "dm" or args.method == "mtt":
        pretrain = get_algorithm("pretrain_dc")
    elif args.method == "kip" or args.method == "frepo":
        pretrain = get_algorithm("pretrain_frepo")
    elif args.method == "krr_st":
        pretrain = get_algorithm("pretrain_krr_st")
    else:
        raise NotImplementedError
    test_algo = get_algorithm("finetune")

    # target_data_name    
    if args.target_data_name == "full":
        if args.source_data_name == "cifar100":
            data_name_list = ["cifar100", "cifar10", "aircraft", "cars", "cub2011", "dogs", "flowers"]
        elif args.source_data_name == "tinyimagenet":
            data_name_list = ["tinyimagenet", "cifar10", "aircraft", "cars", "cub2011", "dogs", "flowers"]
        elif args.source_data_name == "imagenet":
            data_name_list = ["cifar10", "cifar100", "aircraft", "cars", "cub2011", "dogs", "flowers"]
        elif args.source_data_name == "imagenette":
            data_name_list = ["imagenette", "aircraft", "cars", "cub2011", "dogs", "flowers"]
        else:
            raise NotImplementedError
    else:
        data_name_list = args.target_data_name.split("_")        
        
    # train
    acc_dict = { data_name: [] for data_name in data_name_list } 
    for _ in range(args.num_test):
        x_syn, y_syn = x_syn.to(device), y_syn.to(device)
        init_model = pretrain.run(args, device, args.test_model, x_syn, y_syn)
        init_model = init_model.cpu() 
        x_syn, y_syn = x_syn.cpu(), y_syn.cpu()
        for data_name in data_name_list:
            args.num_classes = NUM_CLASSES[data_name]
            if data_name in ["tinyimagenet", "cifar100", "cifar10"]:
                args.test_iteration = 10000
            else:
                args.test_iteration = 5000
            dl_tr, dl_te, aug_tr, aug_te = get_loader(
                args.data_dir, data_name, args.test_batch_size, args.img_size, True)            
            _, acc = test_algo.run(args, device, args.test_model, init_model, dl_tr, dl_te, aug_tr, aug_te)
            print(data_name, acc)
            acc_dict[data_name].append(acc)

    for data_name in data_name_list:
        print(f"{data_name}, mean: {np.mean(acc_dict[data_name])}, std: {np.std(acc_dict[data_name])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # seed
    parser.add_argument('--seed', type=int, default=None)    

    # data
    parser.add_argument('--source_data_name', type=str, default="tinyimagenet")
    parser.add_argument('--target_data_name', type=str, default="full")

    # dir
    parser.add_argument('--data_dir', type=str, default="./datasets")
    parser.add_argument('--synthetic_data_dir', type=str, default="./synthetic_data")
    parser.add_argument('--log_dir', type=str, default="./test_log")

    # dc method
    parser.add_argument('--method', type=str, default="krr_st")

    # hparams for model
    parser.add_argument('--test_model', type=str, default="base")
    parser.add_argument('--dropout', type=float, default=0.0)

    # hparms for test
    parser.add_argument('--num_test', type=int, default=3)

    # gpus
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    # img_size
    if args.source_data_name == "cifar100":
        args.img_size = 32
        if args.test_model == "base":
            args.test_model = "convnet_128_256_512_bn"
    elif args.source_data_name == "tinyimagenet" or args.source_data_name == "imagenet":
        args.img_size = 64
        if args.test_model == "base":
            args.test_model = "convnet_64_128_256_512_bn"
    elif args.source_data_name == "imagenette":
        args.img_size = 224
        if args.test_model == "base":
            args.test_model = "convnet_32_64_128_256_512_bn"
    else:
        raise NotImplementedError
    args.img_shape = (3, args.img_size, args.img_size)

    # pretrain hparams
    if args.method == "random" or args.method == "kmeans" or args.method == "dsa" or args.method == "dm" or args.method == "mtt":
        args.pre_opt = "sgd"
        args.pre_epoch = 1000
        args.pre_iteration = None
        args.pre_batch_size = 256
        args.pre_lr = 0.01
        args.pre_wd = 5e-4

    elif args.method == "kip" or args.method == "frepo":
        #step_per_prototpyes = {10: 1000, 100: 2000, 200: 20000, 400: 5000, 500: 5000, 1000: 10000, 2000: 40000, 5000: 40000}
        args.pre_opt = "adam"
        args.pre_epoch = None
        if args.source_data_name == "cifar100" or args.source_data_name == "imagenet":
            args.pre_iteration = 10000 # 1000            
            args.pre_batch_size = 500
        elif args.source_data_name == "tinyimagenet":
            args.pre_iteration = 40000 # 2000
            if args.test_model == "mobilenet":
                args.pre_batch_size = 256
            else:
                args.pre_batch_size = 500
        elif args.source_data_name == "imagenette":
            args.pre_iteration = 1000 # 10        
            args.pre_batch_size = 10
        args.pre_lr = 0.0003
        args.pre_wd = 0.0

    elif args.method == "krr_st":
        args.pre_opt = "sgd"
        args.pre_epoch = 1000
        args.pre_batch_size = 256
        args.pre_lr = 0.1
        args.pre_wd = 1e-3
    else:
        raise NotImplementedError
    
    # finetune hyperparams
    args.test_opt = "sgd"
    args.test_iteration = None
    if args.img_size == 224 and args.test_model == "resnet18":
        args.test_batch_size = 64
    else:
        args.test_batch_size = 256
    args.test_lr = 0.01
    args.test_wd = 5e-4

    main(args)
