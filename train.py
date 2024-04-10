import random
import argparse
import numpy as np

import torch
import torch.nn as nn

from torchvision.models import resnet18

from data.wrapper import get_loader
from data.augmentation import NUM_CLASSES
from algorithms.wrapper import get_algorithm
from utils import InfIterator, Logger
from model_pool import ModelPool
    
def main(args):
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    # seed
    if args.seed is None:
        args.seed = random.randint(0, 9999)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # data
    args.img_shape = (3, args.img_size, args.img_size)    
    dl, _, _, _ = get_loader(args.data_dir, args.data_name, args.outer_batch_size, args.img_size, False)
    iter_tr = InfIterator(dl)
    dl_tr, dl_te, aug_tr, aug_te = get_loader(args.data_dir, args.data_name, args.test_batch_size, args.img_size, True)

    # target model
    if args.data_name == "imagenette":
        target_model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        target_model.fc = nn.Identity()
        target_model = target_model.to(device)    
    else:
        target_model = resnet18()
        target_model.fc = nn.Identity()
        target_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        target_model.maxpool = nn.Identity()
        target_model = target_model.to(device)      
        target_model.load_state_dict(torch.load(f"./teacher_ckpt/barlow_twins_resnet18_{args.data_name}.pt", map_location="cpu")) 
        
    # get features
    target_model.eval()
    with torch.no_grad():
        x_syn, y_syn = torch.FloatTensor([]), torch.FloatTensor([])
        while x_syn.shape[0] < args.num_images:
            x, y = next(iter_tr)
            x_syn = torch.cat([x_syn, x], dim=0)
            x = x.to(device)
            x = aug_te(x)
            y = target_model(x).cpu()
            torch.cuda.empty_cache() 
            y_syn = torch.cat([y_syn, y], dim=0)

    args.num_pretrain_classes = y_syn.shape[-1]
    args.num_classes = NUM_CLASSES[args.data_name]
    
    # x_syn, y_syn
    x_syn, y_syn = x_syn[:args.num_images].to(device), y_syn[:args.num_images].to(device)
    x_syn.requires_grad_(True); y_syn.requires_grad_(True)
    
    # outer opt
    if args.outer_opt == "sgd":
        outer_opt = torch.optim.SGD([x_syn, y_syn], lr=args.outer_lr, momentum=0.5, weight_decay=args.outer_wd)
    elif args.outer_opt == "adam":
        outer_opt = torch.optim.AdamW([x_syn, y_syn], lr=args.outer_lr, weight_decay=args.outer_wd)
    else:
        raise NotImplementedError
    outer_sch = torch.optim.lr_scheduler.LinearLR(
        outer_opt, start_factor=1.0, end_factor=1e-3, total_iters=args.outer_iteration)    

    # model pool
    model_pool = ModelPool(args, device)
    model_pool.init(x_syn.detach(), y_syn.detach())

    # logger
    logger = Logger(
        save_dir=f"{args.save_dir}/{args.exp_name}",
        save_only_last=True,
        print_every=args.print_every,
        save_every=args.save_every,
        total_step=args.outer_iteration,
        print_to_stdout=True
    )
    logger.register_object_to_save(x_syn, "x_syn")
    logger.register_object_to_save(y_syn, "y_syn")
    logger.start()  

    # algo
    train_algo = get_algorithm("distill")
    pretrain = get_algorithm("pretrain_krr_st")
    test_algo = get_algorithm("linear_eval")

    # outer loop
    for outer_step in range(1, args.outer_iteration+1):
        
        # meta train
        loss = train_algo.run(
            args, device, target_model, model_pool, outer_opt, iter_tr, aug_tr, x_syn, y_syn)
        logger.meter("meta_train", "mse loss", loss)

        # meta test
        if outer_step % args.eval_every == 0 or outer_step == args.outer_iteration:
            init_model = pretrain.run(args, device, args.test_model, x_syn, y_syn)
            loss, acc = test_algo.run(args, device, args.test_model, init_model, dl_tr, dl_te, aug_tr, aug_te)
            del init_model
            logger.meter(f"meta_test", "loss", loss)
            logger.meter(f"meta_test", "accuracy", acc)
            
        outer_sch.step()
        logger.step()
            
    logger.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # seed
    parser.add_argument('--seed', type=int, default=None)

    # data
    parser.add_argument('--target_model', type=str, default="resnet18")
    parser.add_argument('--data_name', type=str, default="cifar100")
    parser.add_argument('--num_workers', type=int, default=0)

    # dir
    parser.add_argument('--data_dir', type=str, default="./datasets")
    parser.add_argument('--save_dir', type=str, default="./results")
    parser.add_argument('--exp_name', type=str, default=None)

    # algorithm
    parser.add_argument('--num_models', type=int, default=10)

    # hparms for online
    parser.add_argument('--online_opt', type=str, default="sgd")
    parser.add_argument('--online_iteration', type=int, default=1000)
    parser.add_argument('--online_lr', type=float, default=0.1)
    parser.add_argument('--online_wd', type=float, default=1e-3)

    # hparms for pretrain
    parser.add_argument('--pre_opt', type=str, default="sgd")
    parser.add_argument('--pre_epoch', type=int, default=1000)
    parser.add_argument('--pre_batch_size', type=int, default=256)
    parser.add_argument('--pre_lr', type=float, default=0.1)
    parser.add_argument('--pre_wd', type=float, default=1e-3)

    # hparms for test
    parser.add_argument('--test_opt', type=str, default="sgd")
    parser.add_argument('--test_iteration', type=int, default=5000)
    parser.add_argument('--test_batch_size', type=float, default=512)
    parser.add_argument('--test_lr', type=float, default=0.2)
    parser.add_argument('--test_wd', type=float, default=0.0)

    # hparms for outer
    parser.add_argument('--outer_opt', type=str, default="adam")
    parser.add_argument('--outer_iteration', type=int, default=160000)
    parser.add_argument('--outer_batch_size', type=int, default=1024)
    parser.add_argument('--outer_lr', type=float, default=1e-3)
    parser.add_argument('--outer_wd', type=float, default=0.)
    parser.add_argument('--outer_grad_norm', type=float, default=0.0)

    # hparams for logger
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=16000)
    parser.add_argument('--save_every', type=int, default=2000)

    # gpus
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    if args.data_name == "cifar100":
        args.img_size = 32
        args.num_images = 1000
        args.train_model = "convnet_128_256_512_bn"
        args.test_model = "convnet_128_256_512_bn"
    elif args.data_name == "tinyimagenet":
        args.img_size = 64
        args.num_images = 2000
        args.train_model = "convnet_64_128_256_512_bn"
        args.test_model = "convnet_64_128_256_512_bn"
    elif args.data_name == "imagenet":
        args.img_size = 64
        args.num_images = 1000
        args.train_model = "convnet_64_128_256_512_bn"
        args.test_model = "convnet_64_128_256_512_bn"
    elif args.data_name == "imagenette":
        args.img_size = 224
        args.num_images = 10
        args.train_model = "convnet_32_64_128_256_512_bn"
        args.test_model = "convnet_32_64_128_256_512_bn"
    else:
        raise NotImplementedError

    main(args)
