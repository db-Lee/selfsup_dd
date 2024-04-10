import torch
import torch.nn.functional as F
from models.wrapper import get_model
from tqdm import trange
from utils import InfIterator


def run(
    args, device, 
    model_name,
    dl_tr, dl_te,
    aug_tr, aug_te
):  
    iter_tr = InfIterator(dl_tr)

    # model
    model = get_model(model_name, args.img_shape, args.num_classes).to(device)

    # opt
    if args.test_opt == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.test_lr, momentum=0.9, weight_decay=args.test_wd)
    elif args.test_opt == "adam":
        opt = torch.optim.AdamW(model.parameters(), lr=args.test_lr, weight_decay=args.test_wd)
    else:
        raise NotImplementedError
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.test_iteration)

    model.train()
    for _ in trange(args.test_iteration):
        x_tr, y_tr = next(iter_tr)
        x_tr, y_tr = x_tr.to(device), y_tr.to(device)
        with torch.no_grad():
            x_tr = aug_tr(x_tr)
        loss = F.cross_entropy(model(x_tr), y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    
    model.eval()
    with torch.no_grad():        
        loss, acc, denominator = 0., 0., 0.
        for x_te, y_te in dl_te:
            x_te, y_te = x_te.to(device), y_te.to(device)
            x_te = aug_te(x_te)
            l_te = model(x_te)
            loss += F.cross_entropy(l_te, y_te, reduction="sum")
            acc += torch.eq(l_te.argmax(dim=-1), y_te).float().sum()
            denominator += x_te.shape[0]
        loss /= denominator; acc /= (denominator/100.)

    del model

    return loss.item(), acc.item()

