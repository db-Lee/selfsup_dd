import torch
import torch.nn.functional as F
from data.augmentation import DiffAugment
from models.wrapper import get_model
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


def distillation_loss(student, teacher, T=1.0, reduction="mean"):
    student = F.log_softmax(student/T, dim=-1)
    teacher = F.softmax(teacher/T, dim=-1)
    loss = -(teacher * student).sum(dim=-1)

    if reduction == "mean":
        return loss.mean(dim=0)
    elif reduction == "sum":
        return loss.sum(dim=0)
    else:
        raise NotImplementedError

def run(
    args, device, 
    model_name, init_model, teacher,
    x_syn, dl_te, aug_te
):  
    
    x_syn = x_syn.detach()
    x_syn = x_syn.to(device)
    dl_syn = DataLoader(TensorDataset(x_syn), batch_size=args.test_batch_size, shuffle=True, num_workers=0)

    # model
    student = get_model(model_name, args.img_shape, args.num_classes).to(device)
    if hasattr(init_model, "fc"):
        del init_model.fc
    student.load_state_dict(init_model.state_dict(), strict=False)

    # opt
    if args.test_opt == "sgd":
        opt = torch.optim.SGD(student.parameters(), lr=args.test_lr, momentum=0.9, weight_decay=args.test_wd)
    elif args.test_opt == "adam":
        opt = torch.optim.AdamW(student.parameters(), lr=args.test_lr, weight_decay=args.test_wd)
    else:
        raise NotImplementedError
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.test_epoch)

    student.train(); teacher.train()
    for _ in trange(args.test_epoch):
        for x, in dl_syn:
            with torch.no_grad():  
                if args.method == "gaussian":
                    x = torch.randn_like(x)
                else:
                    x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)
                y = teacher(x)
            loss = distillation_loss(student(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()    
    
    student.train()
    with torch.no_grad():        
        loss, acc, denominator = 0., 0., 0.
        for x_te, y_te in dl_te:
            x_te, y_te = x_te.to(device), y_te.to(device)
            x_te = aug_te(x_te)
            l_te = student(x_te)
            loss += F.cross_entropy(l_te, y_te, reduction="sum")
            acc += torch.eq(l_te.argmax(dim=-1), y_te).float().sum()
            denominator += x_te.shape[0]
        loss /= denominator; acc /= (denominator/100.)

    del student

    return loss.item(), acc.item()
