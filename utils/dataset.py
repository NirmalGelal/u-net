import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class BasicDataset(Dataset):
    def __init__(self, image_dir, mask_dir, scale=1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.scale = scale
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    @classmethod
    def preprocess(cls, pil_img, scale):
        width, height = pil_img.size
        new_width, new_height = int(scale*width), int(scale*height)

        assert new_width>0 and new_height>0, "Scale is too small"

        pil_img = pil_img.resize((new_width, new_height))

        img_nd = np.array(pil_img)

        # (height, width, channel) to (channel, height, width)
        img_trans = img_nd.transpose((2,0,1))
        # between 0 and 1
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        
        return img_trans
    
    @classmethod
    def preprocess_mask(cls, pil_img, scale):
        width, height = pil_img.size
        new_width, new_height = int(scale*width), int(scale*height)

        assert new_width>0 and new_height>0, "Scale is too small"

        pil_img = pil_img.resize((new_width, new_height))

        img_nd = np.array(pil_img)

        # (height, width, channel) to (channel, height, width)
        img_trans = img_nd.transpose((2,0,1))
        # torch.set_printoptions(edgeitems=10)
        
        return img_trans
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("sat.jpg","mask.png"))

        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))

        assert image.size == mask.size, f'image and mask size must be same size'

        image = self.preprocess(image, self.scale)
        mask = self.preprocess_mask(image, self.scale)
        # need to call one hot encoding of labels which is yet to implement

        return {
            'image': torch.from_numpy(image).type(torch.FloatTensor),
            'mask': mask
        }



if __name__ == "__main__":
    dataset = BasicDataset(r"data\training_data\images", r"data\training_data\masks")
    print(f'length of dataset => {len(dataset)}')
    print(dataset.__getitem__(234))