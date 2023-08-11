import torch
import torch.nn.functional as F
from tqdm import tqdm

def eval_net(net, loader, device):

    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)
    total=0
    
    with tqdm(total=n_val, unit='batch', disable=True, leave=True) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            
            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                total+=F.cross_entropy(mask_pred, true_masks).item()


            pbar.update()

    net.train()
    return total/n_val