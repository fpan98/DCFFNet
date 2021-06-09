import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import cv2


class Dataset(Dataset):

    def __init__(self, base_path, txt_path, LEVIR=False):

        self.base_path = base_path
        self.img_txt_path = txt_path
        self.imgs_path_list = np.loadtxt(self.img_txt_path, dtype=str)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 从imagenet数据集上抽样得到的均值和方差
        ])
        self.gt_transformer = transforms.Compose([
            transforms.ToTensor(),  # (H*W*C)->(C*H*W) and range[0,255]->range[0.0,1.0]
        ])
        self.img_label_path_pairs = self.get_img_label_path_pairs()

        self.LEVIR = LEVIR  # LEVIR中的gt是三通的的RGB图

    def get_img_label_path_pairs(self):
        global image1_name, image2_name
        img_label_pair_list = {}
        for idx, did in enumerate(open(self.img_txt_path)):
            try:
                image1_name, image2_name, mask_name = did.strip("\n").split(' ')
            except ValueError:  # Adhoc for test.
                image_name = mask_name = did.strip("\n")
            img1_file = os.path.join(self.base_path, image1_name)
            img2_file = os.path.join(self.base_path, image2_name)
            lbl_file = os.path.join(self.base_path, mask_name)
            img_label_pair_list.setdefault(idx, [img1_file, img2_file, lbl_file])
        return img_label_pair_list

    def __getitem__(self, index):
        img1_path, img2_path, label_path = self.img_label_path_pairs[index]
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        img1 = np.array(img1, dtype=np.uint8)
        img2 = np.array(img2, dtype=np.uint8)
        if self.LEVIR:
            label = Image.open(label_path).convert('L')
        else:
            label = Image.open(label_path)
        label = np.array(label, dtype=np.uint8)

        label_d2 = cv2.resize(label, (label.shape[0]//2, label.shape[1]//2))
        label_d4 = cv2.resize(label, (label.shape[0]//4, label.shape[1]//4))
        label_d8 = cv2.resize(label, (label.shape[0]//8, label.shape[1]//8))
        label_d16 = cv2.resize(label, (label.shape[0]//16, label.shape[1]//16))

        img1 = self.image_transform(img1)  # [-1.0,1.0]
        img2 = self.image_transform(img2)

        label = self.gt_transformer(label)  # [0.0, 1.0]
        label_d2 = self.gt_transformer(label_d2)
        label_d4 = self.gt_transformer(label_d4)
        label_d8 = self.gt_transformer(label_d8)
        label_d16 = self.gt_transformer(label_d16)

        # return img1, img2, label, label_d2, label_d4, label_d8, label_d16
        return img1, img2, label

    def __len__(self):
        return len(self.img_label_path_pairs)

