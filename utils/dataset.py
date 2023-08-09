import os
import logging
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class BasicDataset(Dataset):
    def __init__(self, image_dir, mask_dir, scale=1, mask_suffix=''):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.scale = scale


        self.ids = [os.path.splitext(file)[0] for file in os.listdir(image_dir) if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)
    
    @classmethod
    def preprocess(cls, pil_img, scale):
        # print(pil_img)
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

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # (height, width, channel) to (channel, height, width)
        img_trans = img_nd.transpose((2,0,1))
        torch.set_printoptions(edgeitems=10)
        
        return img_trans
    
    @classmethod
    def RGB_2_class_idx(cls, mask_to_be_converted):
        mapping = {(0  , 255, 255): 0,    #urban_land
                (255, 255, 0  ): 1,    #agriculture
                (255, 0  , 255): 2,    #range_land
                (0  , 255, 0  ): 3,    #forest_land
                (0  , 0  , 255): 4,    #water
                (255, 255, 255):5,     #barren_land
                (0  , 0  , 0  ):6}     #unknown
        
        # create numpy array of as mask_img
        temp = np.array(mask_to_be_converted)

        # set threshold value of 128. If pixel is less than 128 set it to 0 else 1.
        temp = np.where(temp>=128, 255, 0)

        class_mask=torch.from_numpy(temp)
        h, w = class_mask.shape[1], class_mask.shape[2]

        # create same sized empty array
        mask_out = torch.zeros(h, w, dtype=torch.long)
            
        for k in mapping:
            idx = (class_mask == torch.tensor(k,dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3)
            mask_out[validx] = torch.tensor(mapping[k], dtype=torch.long)
        return mask_out

    def __getitem__(self, index):
        idx = self.ids[index]
        idx = idx[:len(idx) - 4]
        
        mask_file = glob(self.mask_dir + r'\\' + idx + '_mask' + '.png')
        img_file = glob(self.image_dir + r'\\' + idx + '_sat' + '.jpg')

        assert len(mask_file) == 1, f'Either no mask or multiple mask found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, f'Either no mask or multiple mask found for the ID {idx}: {img_file}'

        image = Image.open(img_file[0])
        mask = Image.open(mask_file[0])

        assert image.size == mask.size, f'image and mask size must be same size'

        image = self.preprocess(image, self.scale)
        mask = self.preprocess_mask(mask, self.scale)
        mask = self.RGB_2_class_idx(mask)

        return {
            'image': torch.from_numpy(image).type(torch.FloatTensor),
            'mask': mask
        }



if __name__ == "__main__":
    dataset = BasicDataset(r"data\training_data\images", r"data\training_data\masks")
    print(f'length of dataset => {len(dataset)}')
    print(dataset.__getitem__(178))