import torch
from data.augmentation import DiffAugment
from models.wrapper import get_model
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from transformers import get_cosine_schedule_with_warmup
from utils import InfIterator


def run(
    args, device, 
    model_name, 
    x_syn, y_syn
):  
    x_syn, y_syn = x_syn.detach(), y_syn.detach()
    x_syn, y_syn = x_syn.to(device), y_syn.to(device)
    dl_syn = DataLoader(TensorDataset(x_syn, y_syn), batch_size=args.pre_batch_size, shuffle=True, num_workers=0)
    iter_syn = InfIterator(dl_syn)
    
    # model and opt
    model = get_model(model_name, args.img_shape, args.num_pretrain_classes).to(device)
    if args.pre_opt == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.pre_lr, momentum=0.9, weight_decay=args.pre_wd)
    elif args.pre_opt == "adam":
        opt = torch.optim.AdamW(model.parameters(), lr=args.pre_lr, weight_decay=args.pre_wd)
    else:
        raise NotImplementedError
    sch = get_cosine_schedule_with_warmup(opt, 500, args.pre_iteration)
    
    # pretrain        
    model.train()
    for _ in trange(1, args.pre_iteration+1):
        x_syn, y_syn = next(iter_syn)
        # loss
        with torch.no_grad():
            x_syn = DiffAugment(x_syn, args.dsa_strategy, param=args.dsa_param)            
        loss = torch.mean(torch.sum((model(x_syn) - y_syn) ** 2 * 0.5, dim=-1))
        # update
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()

    return model
            


            

