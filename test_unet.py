from unet.unet_model import UNet
import cv2
from PIL import Image
from torchvision.transforms import transforms
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
import torch

img = Image.open(r'random_selected_data\training_data\images\266_sat.jpg').convert("RGB")

image_dir = r'data1\training_data\images'
mask_dir = r'data1\training_data\masks'
dataset = BasicDataset(image_dir, mask_dir)



n_val = int(len(dataset) * 0.1)
n_train = len(dataset) - n_val
train, val = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train, batch_size=1)
net = UNet(n_channels=3, n_classes=7)
for batch in train_loader:

    imgs = batch['image']
    print(imgs)
    true_masks = batch['mask']
    # print(true_masks)

    masks_pred = net(imgs)
    print(masks_pred)

# print(img.size)

# net = UNet()
