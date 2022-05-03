"""Datasets"""

import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import glob
import PIL
import imageio
import random
import math
import pickle
import numpy as np
import torch.nn.functional as F
#####

class CelebA(Dataset):
    """CelelebA Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.Resize(320), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((img_size, img_size), interpolation=0)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0

class Cats(Dataset):
    """Cats Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()
        
        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=0), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.RandomHorizontalFlip(p=0.5)])
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        
        return X, 0

class Carla(Dataset):
    """Carla Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()
        
        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=0), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        
        return X, 0

class DVRDataset(Dataset):
    """
    Dataset from DVR (Niemeyer et al. 2020)
    Provides 3D-R2N2 and NMR renderings
    """
    def __init__(
        self,
        dataset_path,
        stage="train",
        list_prefix="softras_", 
        img_size=None,
        scale_focal=True,
        max_imgs=100000,
        **kwargs
    ):
        """
        :Original Setting : z_near=1.2, z_far=4.0,
        :param path dataset root path, contains metadata.yml
        :param stage train | val | test
        :param list_prefix prefix for split lists: <list_prefix>[train, val, test].lst
        :param image_size result image size (resizes if different); None to keep original size
        :param scale_focal if true, assume focal length is specified for
        image of side length 2 instead of actual image size. This is used
        where image coordinates are placed in [-1, 1].
        """
        super().__init__()
        self.base_path = dataset_path
        assert os.path.exists(self.base_path)

        cats = [x for x in glob.glob(os.path.join(dataset_path, "*")) if os.path.isdir(x)]

        if stage == "train":
            file_lists = [os.path.join(x, list_prefix + "train.lst") for x in cats]
        elif stage == "val":
            file_lists = [os.path.join(x, list_prefix + "val.lst") for x in cats]
        elif stage == "test":
            file_lists = [os.path.join(x, list_prefix + "test.lst") for x in cats]

        all_objs = []
        for file_list in file_lists:
            if not os.path.exists(file_list):
                continue
            base_dir = os.path.dirname(file_list)
            cat = os.path.basename(base_dir)
            with open(file_list, "r") as f:
                objs = [(cat, os.path.join(base_dir, x.strip())) for x in f.readlines()]
            all_objs.extend(objs)

        self.all_objs = all_objs

        #### NEW
        all_imgs = []
        for _, root_dir in self.all_objs:
            tmp = [
                x
                for x in glob.glob(os.path.join(root_dir, "image", "*"))
                if (x.endswith(".jpg") or x.endswith(".png"))
            ]
            all_imgs.extend(tmp)
        self.all_imgs = sorted(all_imgs)

        ####
        self.stage = stage
        
        print(
            "Loading DVR dataset",
            self.base_path,
            "stage",
            stage,
            len(self.all_objs),
            "objs",
            "shapenet"
        )

        self.image_size = img_size

        ### Shapenet only
        self._coord_trans_world = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        self._coord_trans_cam = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        self.scale_focal = scale_focal
        self.max_imgs = max_imgs

        self.lindisp = False
        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=0), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])


    def __len__(self):
        return len(self.all_objs)

    def __getitem__(self, index):
        # GetIndex
        X = PIL.Image.open(self.all_imgs[index])
        X = self.transform(X)
        return X, 0

"""
        if len(rgb_paths) <= self.max_imgs:
            sel_indices = np.arange(len(rgb_paths))
        else:
            sel_indices = np.random.choice(len(rgb_paths), self.max_imgs, replace=False)
            rgb_paths = [rgb_paths[i] for i in sel_indices]

        cam_path = os.path.join(root_dir, "cameras.npz")
        all_cam = np.load(cam_path)

        all_imgs = []
        all_poses = []

        focal = None

        for idx, (rgb_path) in enumerate(zip(rgb_paths)):
            i = sel_indices[idx]
            img = imageio.imread(rgb_path)[..., :3]

            if self.scale_focal:
                x_scale = img.shape[1] / 2.0
                y_scale = img.shape[0] / 2.0
                xy_delta = 1.0
            else:
                x_scale = y_scale = 1.0
                xy_delta = 0.0

            # ShapeNet
            wmat_inv_key = "world_mat_inv_" + str(i)
            wmat_key = "world_mat_" + str(i)
            if wmat_inv_key in all_cam:
                extr_inv_mtx = all_cam[wmat_inv_key]
            else:
                extr_inv_mtx = all_cam[wmat_key]
                if extr_inv_mtx.shape[0] == 3:
                    extr_inv_mtx = np.vstack((extr_inv_mtx, np.array([0, 0, 0, 1])))
                extr_inv_mtx = np.linalg.inv(extr_inv_mtx)

            intr_mtx = all_cam["camera_mat_" + str(i)]
            fx, fy = intr_mtx[0, 0], intr_mtx[1, 1]
            assert abs(fx - fy) < 1e-9

            fx = fx * x_scale

            if focal is None:
           
        return
        """


def get_dataset(name, subsample=None, batch_size=1, **kwargs):
    dataset = globals()[name](**kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=8
    )
    return dataloader, 3

def get_dataset_distributed(name, world_size, rank, batch_size, **kwargs):
    # Used in Training
    if name.startswith("dvr"):
        dataset = DVRDataset(stage='train', **kwargs)
    else:
        dataset = globals()[name](**kwargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )

    return dataloader, 3
