from PIL import Image
from data.base_dataset import BaseDataset, make_transform
from util.info_generation import generate_info, generate_class_map
import os.path as osp
import torch
import numpy as np
import random

class SourceOnlyDataset(BaseDataset):
    
    def __init__(self, opt):
        """
        Initialization, generate sample information and load them into memory
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.isTrain = opt.phase == 'training'
        self.root = osp.join(opt.dataroot, opt.phase)
        self.infos = generate_info(self.root) 
        if not self.isTrain:
            self.infos *= self.opt.batch_size
        self.image_transform = make_transform(opt, Image.BICUBIC, toTensor=True)
        self.label_transform = make_transform(opt, Image.NEAREST, isLabel=True)
        self.label_class_map = generate_class_map(opt.dataroot, name=opt.mapping_file_name)
    
    def __getitem__(self, index):
        # geneate random seed
        seed = np.random.randint(2147483647)

        # get infomation of both image and label
        info = self.infos[index]
        image_path = info['image_path']
        label_path = info['label_path']
        image_name = info['image_name']
        assert osp.exists(image_path)
        assert osp.exists(label_path)
        
        # load image and label in memory
        label = Image.open(label_path)
        image = Image.open(image_path).convert('RGB')
        
        # NOTE map current categories to new ones
        label = np.asarray(label, np.uint8)
        def f(e): return self.label_class_map[e]["newid"]
        vf = np.vectorize(f)
        label = vf(label)
        label = label.astype(np.uint8)
        # preprocessing
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.image_transform(image)
        random.seed(seed)
        torch.manual_seed(seed)
        label = self.label_transform(label)

        # label posprocessing
        label = torch.tensor(np.asarray(label), dtype=torch.float32)
        
        return {'image':image, 'label': label, 'name':image_name}

    def __len__(self):
        return len(self.infos)

    def get_classes(self):
        return {info["newid"]:info["newname"] for info in self.label_class_map.values()}
    



        
    
