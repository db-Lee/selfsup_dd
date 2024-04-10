import random
import argparse
import numpy as np

import torch

from data.wrapper import get_loader
from data.augmentation import NUM_CLASSES
from algorithms.wrapper import get_algorithm
    
def main(args):
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    # seed
    if args.seed is None:
        args.seed = random.randint(0, 9999)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(args)

    # algo
    test_algo = get_algorithm("scratch")

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
        for data_name in data_name_list:
            args.num_classes = NUM_CLASSES[data_name]
            if data_name in ["tinyimagenet", "cifar100", "cifar10"]:
                args.test_iteration = 10000
            else:
                args.test_iteration = 5000
            dl_tr, dl_te, aug_tr, aug_te = get_loader(
                args.data_dir, data_name, args.test_batch_size, args.img_size, True)            
            _, acc = test_algo.run(args, device, args.test_model, dl_tr, dl_te, aug_tr, aug_te)
            print(data_name, acc)
            acc_dict[data_name].append(acc)

    for data_name in data_name_list:
        print(f"{data_name}, mean: {np.mean(acc_dict[data_name])}, std: {np.std(acc_dict[data_name])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # seed
    parser.add_argument('--seed', type=int, default=None)    

    # data
    parser.add_argument('--source_data_name', type=str, default="cifar100")
    parser.add_argument('--target_data_name', type=str, default="full")

    # dir
    parser.add_argument('--data_dir', type=str, default="../evaluation_seanie/datasets")
    parser.add_argument('--synthetic_data_dir', type=str, default="./synthetic_data")
    parser.add_argument('--log_dir', type=str, default="./test_log")

    # hparams for model
    parser.add_argument('--test_model', type=str, default="base")

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
