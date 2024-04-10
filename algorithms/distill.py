import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def run(
    args, device, target_model,
    model_pool, outer_opt,
    iter_tr, aug,
    x_syn, y_syn
):
    outer_opt.zero_grad()

    # prepare model
    idx = np.random.randint(args.num_models)
    model = model_pool.models[idx]
    model.eval(); model.zero_grad()

    # data
    x_real, _ = next(iter_tr)
    x_real = x_real.to(device)
    target_model.eval()
    with torch.no_grad():
        x_real = aug(x_real)
        y_real = target_model(x_real)
        torch.cuda.empty_cache() 

    # feature
    f_syn = model.embed(x_syn)
    f_syn = torch.cat([ f_syn, torch.ones(f_syn.shape[0], 1).to(device) ], dim=1)
    with torch.no_grad():
        f_real = model.embed(x_real)
        f_real = torch.cat([ f_real, torch.ones(f_real.shape[0], 1).to(device) ], dim=1)

    # kernel
    K_real_syn = f_real @ f_syn.permute(1, 0).contiguous()
    K_syn_syn = f_syn @ f_syn.permute(1, 0).contiguous()

    # lambda and eye
    lambda_ = 1e-6 * torch.trace(K_syn_syn.detach())
    eye = torch.eye(K_syn_syn.shape[0]).to(device)

    # mse loss
    outer_loss = F.mse_loss(
        K_real_syn @ torch.linalg.solve(K_syn_syn + lambda_*eye, y_syn),
        y_real
    )
    outer_grad = torch.autograd.grad(outer_loss, [x_syn, y_syn])

    # meta update    
    if x_syn.grad is None:
        x_syn.grad = outer_grad[0]
    else:
        x_syn.grad.data.copy_(outer_grad[0].data)
    if y_syn.grad is None:
        y_syn.grad = outer_grad[1]
    else:
        y_syn.grad.data.copy_(outer_grad[1].data)
    if args.outer_grad_norm > 0.:
        nn.utils.clip_grad_norm_(x_syn, args.outer_grad_norm)
        nn.utils.clip_grad_norm_(y_syn, args.outer_grad_norm)
    outer_opt.step()

    model_pool.update(idx, x_syn.detach(), y_syn.detach())

    return outer_loss
