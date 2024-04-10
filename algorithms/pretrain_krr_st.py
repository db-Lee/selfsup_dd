import torch
import torch.nn.functional as F
from models.wrapper import get_model
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


def run(
    args, device, 
    model_name, 
    x_syn, y_syn
):  
    x_syn, y_syn = x_syn.detach(), y_syn.detach()
    x_syn, y_syn = x_syn.to(device), y_syn.to(device)
    dl_syn = DataLoader(TensorDataset(x_syn, y_syn), batch_size=args.pre_batch_size, shuffle=True, num_workers=0)

    # model and opt
    model = get_model(model_name, args.img_shape, args.num_pretrain_classes).to(device)
    if args.pre_opt == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.pre_lr, momentum=0.9, weight_decay=args.pre_wd)
    elif args.pre_opt == "adam":
        opt = torch.optim.AdamW(model.parameters(), lr=args.pre_lr, weight_decay=args.pre_wd)
    else:
        raise NotImplementedError
    
    # pretrain        
    model.train()
    for _ in trange(1, args.pre_epoch+1):
        for x_syn, y_syn in dl_syn:
            # loss
            loss = F.mse_loss(model(x_syn), y_syn)
            # update
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model

            

