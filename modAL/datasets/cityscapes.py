import glob
import os.path as osp
from typing import Optional, Union

from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset


class CityscapesDataset(Dataset):
    """cityscapes semantic segmentation dataset

    root_dir
    ├── gtFine
    │   ├── test
    │   ├── train
    │   └── val
    └── leftImg8bit
        ├── test
        ├── train
        └── val
    """

    def __init__(self,
                 config: Union[DictConfig, ListConfig],
                 split: str,
                 ann_dir: str = 'gtFine',
                 img_dir: str = 'leftImg8bit',
                 img_suffix: str = '_leftImg8bit.png',
                 seg_map_suffix: str = '_gtFine_labelTrainIds.png'):
        self.config = config
        # class_id: class_name
        self.class_names = config.data.class_names
        self.root_dir = config.data.root_dir
        self.split = split

        assert split in ['train', 'val', 'test']

        self.img_paths = glob.glob(osp.join(self.root_dir, img_dir, split, '**', '*' + img_suffix), recursive=True)

        if len(self.img_paths) == 0:
            print(f'not find image under {self.root_dir} {img_dir} with {img_suffix}')

        assert len(self.img_paths) > 0, f'not find image under {self.root_dir} {img_dir} with {img_suffix}'

        self.lbl_paths = glob.glob(osp.join(self.root_dir, ann_dir, split, '**', '*' + seg_map_suffix), recursive=True)

        assert len(self.lbl_paths) == len(self.img_paths)

        # pair up img_paths[i] with lbl_paths[i]
        self.img_paths.sort()
        self.lbl_paths.sort()

    def __getitem__(self, index):
        return self.img_paths[index], self.lbl_paths[index]

    def __len__(self):
        return len(self.img_paths)
