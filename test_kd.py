import argparse
import random

import numpy as np
import torch

from algorithms.wrapper import get_algorithm
from data.augmentation import NUM_CLASSES, ParamDiffAug
from data.wrapper import get_loader
from models.wrapper import get_model


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
    if args.method != "gaussian":
        x_syn = torch.load(f"./synthetic_data/{args.source_data_name}/{args.method}/x_syn.pt", map_location="cpu").detach()
        y_syn = torch.load(f"./synthetic_data/{args.source_data_name}/{args.method}/y_syn.pt", map_location="cpu").detach()
        if args.method == "kip" or args.method == "frepo" or args.method == "krr_st":
            y_syn = y_syn.float()
        else:
            y_syn = y_syn.long()
    else:
        x_syn = torch.load(f"./synthetic_data/{args.source_data_name}/random/x_syn.pt", map_location="cpu").detach()
        y_syn = torch.load(f"./synthetic_data/{args.source_data_name}/random/y_syn.pt", map_location="cpu").detach()
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
    elif args.method == "gaussian":
        pass
    else:
        raise NotImplementedError
    test_algo = get_algorithm("zeroshot_kd")  
    
    data_name = "cifar10"
    args.img_shape = (3, args.img_size, args.img_size)
    dl_tr, dl_te, aug_tr, aug_te = get_loader(
        args.data_dir, data_name, args.test_batch_size, args.img_size, True)
    data = {
        "num_classes": NUM_CLASSES[data_name.lower()],
        "dl_tr": dl_tr,
        "dl_te": dl_te,
        "aug_tr": aug_tr,
        "aug_te": aug_te
    }

    teacher = get_model("resnet18", args.img_shape, data["num_classes"]).to(device)
    ckpt = torch.load(f"./teacher_ckpt/teacher_{data_name}.pt", map_location="cpu")
    teacher.load_state_dict(ckpt)
    for p in teacher.parameters():
        p.requires_grad_(False)

    acc_list = []
    for _ in range(args.num_test):
        if args.method != "gaussian":
            init_model = pretrain.run(args, device, args.test_model, x_syn, y_syn)
        else:
            init_model = get_model(args.test_model, args.img_shape, 1).to(device)
                                         
        args.num_classes = data["num_classes"]
        dl_te = data["dl_te"]
        aug_te = data["aug_te"]
        _, acc = test_algo.run(args, device, args.test_model, init_model, teacher, x_syn, dl_te, aug_te)
        print(acc)
        acc_list.append(acc)

    print(f"{data_name}, mean: {np.mean(acc_list)}, std: {np.std(acc_list)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # seed
    parser.add_argument('--seed', type=int, default=None)    

    # data
    parser.add_argument('--source_data_name', type=str, default="tinyimagenet")
    parser.add_argument('--target_data_name', type=str, default="cifar10")

    # dir
    parser.add_argument('--data_dir', type=str, default="../evaluation_seanie/datasets")
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
    if args.method == "gaussian" or args.method == "random" or args.method == "kmeans" or args.method == "dsa" or args.method == "dm" or args.method == "mtt":
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
                args.pre_batch_size = 128
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
    
    # zeroshot_kd hyperparams
    args.test_opt = "adam"
    args.test_epoch = 1000
    args.test_batch_size = 512
    if args.method == "gaussian":
        args.test_lr = 1e-3
    else:
        args.test_lr = 1e-4
    args.test_wd = 0.

    main(args)
