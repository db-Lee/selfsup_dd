import torch
from torch.utils.data import TensorDataset, DataLoader

from data.augmentation import get_aug

def get_loader(root_dir, data_name, batch_size, size=32, aug=False):
    X_tr, Y_tr = torch.load(f"{root_dir}/{data_name}/X_tr_{size}.pth"), torch.load(f"{root_dir}/{data_name}/Y_tr_{size}.pth")
    dataset_tr = TensorDataset(X_tr, Y_tr)
    dataloader_tr = DataLoader(dataset_tr, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True)

    if data_name == "imagenet":        
        dataloader_te = dataloader_tr
    else:
        X_te, Y_te = torch.load(f"{root_dir}/{data_name}/X_te_{size}.pth"), torch.load(f"{root_dir}/{data_name}/Y_te_{size}.pth")    
        dataset_te = TensorDataset(X_te, Y_te)  
        dataloader_te = DataLoader(dataset_te, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True)   
    

    transform_tr, transform_te = get_aug(data_name, size, aug)

    return dataloader_tr, dataloader_te, transform_tr, transform_te