import logging
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.dataset import BasicDataset
from eval import eval_net
from torch.utils.data import DataLoader, random_split
from classcount import classcount
from unet.unet_model import UNet

torch.autograd.set_detect_anomaly(True)

image_dir = r".\data1\training_data\images"
mask_dir = r".\data1\training_data\masks"

# dataset = BasicDataset(image_dir, mask_dir, 0.5)
checkpoint_dir = 'checkpoints/'

def train_net(net, device, epochs=5, batch_size=1, lr=0.001, val_percent=0.1, save_cp=True, img_scale=0.5):

    dataset = BasicDataset(image_dir, mask_dir, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader =DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')

    logging.info(f'''Starting training:
                Epochs:             {epochs}
                Batch size:         {batch_size}
                Learning rate:      {lr}
                Training size:      {n_train}
                Validation size:    {n_val}
                Checkpoints:        {save_cp}
                Device:             {device.type}
                Images scaling:     {img_scale}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)

    weights_classes = torch.from_numpy(classcount(train_loader))
    weights_classes = weights_classes.to(device=device, dtype=torch.float32)

    print("Class Distribution: ", weights_classes)

    criterion = nn.CrossEntropyLoss(weight=weights_classes)

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch+1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # net.half()

                imgs = batch['image']
                true_masks = batch['mask']

                # net = net.to(device=device)

                imgs = imgs.to(device=device, dtype=torch.float16)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)

                masks_pred = masks_pred.type(torch.float32)

                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                pbar.set_postfix(**{'Epoch Loss': epoch_loss/n_train})

                # net.float()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])


                val_score = eval_net(net, val_loader, device)

                logging.info('Validation CE Loss: {}'.format(val_score))
                writer.add_scalar('Loss/test', val_score, epoch+1)
                writer.add_images('images', imgs, epoch+1)

                if (epoch+1)%5 == 0:
                    if save_cp:
                        try:
                            os.mkdir(checkpoint_dir)
                            logging.info('Created checkpoint directory')
                        except OSError:
                            pass
                        torch.save(net.state_dict(), checkpoint_dir + f'CP_epoch {epoch+1}.pth')
                        logging.info(f'Checkpoint {epoch+1} saved !')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=3, n_classes=7, bilinear=True)
    train_net(
        net=net,
        epochs=5,
        batch_size=2,
        lr=1e-5,
        device=device,
        img_scale=0.5,
        val_percent=0.2
    )