import logging
import os

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
import classcount


image_dir = r"data\training_data\images"
mask_dir = r"data\training_data\masks"

def train_net(net, device, epochs=5, batch_size=1, lr=0.001, test_size=0.1, img_scale=0.5):

    dataset = BasicDataset(image_dir, mask_dir, img_scale)
    n_test = int(len(dataset) * test_size)
    n_train = len(dataset) - n_test
    train, test = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader =DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    logging.info(f'''Starting training:
                Epochs:         {epochs}
                Batch size:     {batch_size}
                Learning rate:  {lr}
                Training size:  {n_train}
                Testing size:   {n_test}
                Device:         {device.type}
                Images scaling: {img_scale}
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
                net.half()

                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float16)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                masks_pred = net(imgs)

                masks_pred = masks_pred.type(torch.float32)

                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                pbar.set_postfix({'Epoch Loss': epoch_loss/n_train})

                net.float()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])

                # test_score = eval_net(net, test_loader, device)
