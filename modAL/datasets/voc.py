import glob
import os.path as osp
from typing import Union

from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    """VOC detection dataset, with yolov5 format

    root_dir
    ├── images
    │   ├── test2007
    │   ├── train2007
    │   ├── train2012
    │   ├── val2007
    │   └── val2012
    ├── labels
    │   ├── test2007
    │   ├── train2007
    │   ├── train2012
    │   ├── val2007
    │   └── val2012
    """

    def __init__(self, config: Union[DictConfig, ListConfig], split: str):
        self.config = config
        # class_id: class_name
        self.class_maps = config.data.names
        self.root_dir = config.data.path
        self.split = split

        if self.split in ['train', 'val']:
            years = [2007, 2012]
        else:
            years = [2007]

        self.img_paths = []
        for year in years:
            self.img_paths += glob.glob(osp.join(self.root_dir, f'images/{self.split}{year}/*.jpg'))

        self.lbl_paths = [self.imgpath_to_label(img_path) for img_path in self.img_paths]

        N = len(self.img_paths)
        assert N > 0, f'no images found under {self.root_dir}'
        print(f'find {N} image and labels')

    def imgpath_to_label(self, img_path):
        img_rel_path = osp.relpath(img_path, start=self.root_dir)
        lbl_rel_path = img_rel_path.replace('images', 'labels').replace('.jpg', '.txt')
        label_path = osp.join(self.root_dir, lbl_rel_path)
        return label_path

    def __getitem__(self, index):
        return self.img_paths[index], self.lbl_paths[index]

    def __len__(self):
        return len(self.img_paths)
