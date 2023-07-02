# -*- coding: UTF-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torchvision.transforms as T

from .base_model import BaseModel
from . import networks
from .losses import  BNSLoss, MaxSqaureLoss
from .network_bank.sfda import ChannelAttentionMap, SpatialAttentionMap, DAM


class SFDAModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--init_path', type=str, required=True)
        parser.add_argument('--lr_G', type=float, default=0.1)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # TODO visualization settings
        self.loss_names = ['BNS', 'MAE', 'ssDAD', 'stDAD', 'TAR']
        self.visual_names = [
            'rgb',
            'color_gt',
            'color_pred_source',
            'color_pred_target'
        ]
        if not self.isTrain:
            self.visual_names += ['gt', 'mask']

        # network definition and initialization
        self.model_names = ['G', 'S', 'T']
        self.netG = networks.define_network(256, 3, 'dcgan_generator', gpu_ids=self.gpu_ids)
        self.netS = networks.define_network(opt.input_nc, opt.output_nc, netTask=opt.netTask, gpu_ids=self.gpu_ids, pretrained=False)
        self.netT = networks.define_network(opt.input_nc, opt.output_nc, netTask=opt.netTask, gpu_ids=self.gpu_ids, pretrained=False)
        
        if self.isTrain:
        # model initialization
            path = opt.init_path
            print("Init model from %s " %  path)
            state_dict = torch.load(path, map_location=str(self.device))
            self.netS.module.load_state_dict(state_dict)
            self.set_requires_grad(self.netS, False)
            self.netT.module.load_state_dict(state_dict)
        
        # define attention module
            self.CAM = ChannelAttentionMap(512)
            self.SAM = SpatialAttentionMap(512)
            self.DAM = DAM(512)

        # define loss functions and optimizers
            self.criterion_BNS = BNSLoss(self.netS) # should be defined after initializing netS as BNS needs to be calculated
            self.criterion_L1 = nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_G)
            self.optimizer_T = torch.optim.SGD(self.netT.parameters(), lr=opt.lr)
        
    def set_input(self, data, isTrain=None):
        self.rgb = data['image'].to(self.device)
        self.gt = data['label'].to(self.device, dtype=torch.long)
        self.image_paths = data['name']

    def foward(self):



        pass

    def generate_fake_sample(self):
        # generate fake sample
        noise = torch.randn(self.rgb.size(0), 256, 1, 1, device=self.device)
        self.rgb_fake = self.netG(noise)

    def forward_source_fake(self):
        with torch.no_grad():
            self.pred_source_fake = self.netS(self.rgb_fake)
    
    def forward_target_fake(self):
        self.pred_target_fake = self.netT(self.rgb)

    def cal_BNS(self):
        self.loss_BNS = self.criterion_BNS(self.netT)

    def cal_MAE(self):
        self.loss_MAE = self.criterion_L1(self.pred_target_fake, self.pred_source_fake)
        
    def cal_ssDAD(self):
        # get features of fake sample
        features_source_fake = self.netS.module.get_features()
        features_target_fake = self.netT.module.get_features()

        # get attention amp
        attention_source = self.DAM(features_source_fake)
        attention_target = self.DAM(features_target_fake)

        # calculate ssDAD
        self.loss_ssDAD(attention_target, attention_source)

    def cal_stDAD(self):
        # get 

    def compuite_visuals(self):
        with torch.no_grad():
            pass

    def bacward_G(self):
        pass

    

