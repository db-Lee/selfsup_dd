import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from models.wrapper import get_model


class ModelPool:
    def __init__(self, args, device):
        self.device = device        
        self.online_iteration = args.online_iteration
        self.num_models = args.num_models

        # model func
        self.model_func = lambda _: get_model(args.train_model, args.img_shape, args.num_pretrain_classes).to(device)

        # opt func
        if args.online_opt == "sgd":
            self.opt_func = lambda param: torch.optim.SGD(param, lr=args.online_lr, momentum=0.9, weight_decay=args.online_wd)
        elif args.online_opt == "adam":
            self.opt_func = lambda param: torch.optim.AdamW(param, lr=args.online_lr, weight_decay=args.online_wd)
        else:
            raise NotImplementedError
        
        self.iterations = [ 0 ] *self.num_models
        self.models = [ self.model_func(None) for _ in range(self.num_models) ]
        self.opts = [ self.opt_func(self.models[i].parameters()) for i in range(self.num_models) ]

    def init(self, x_syn, y_syn):
        for idx in range(self.num_models):
            online_iteration = np.random.randint(1, self.online_iteration)
            self.iterations[idx] = online_iteration
            model = self.models[idx]
            opt = self.opts[idx]
            model.train()
            print(f"{idx}-th model init")
            for _ in trange(online_iteration):
                opt.zero_grad()
                loss = F.mse_loss(model(x_syn), y_syn)
                loss.backward()
                opt.step()

    def update(self, idx, x_syn, y_syn):
        # reset
        if self.iterations[idx] >= self.online_iteration:
            self.models[idx] = self.model_func(None)
            self.opts[idx] = self.opt_func(self.models[idx].parameters())            
            model = self.models[idx]
            opt = self.opts[idx]

        # train the model for 1 step
        else:
            self.iterations[idx] = self.iterations[idx] + 1
            model = self.models[idx]
            opt = self.opts[idx]

            model.train()
            opt.zero_grad()
            loss = F.mse_loss(model(x_syn), y_syn)
            loss.backward()
            opt.step()
