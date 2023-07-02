# -*- coding: UTF-8 -*-
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T

class Deeplabv3Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # visualization settings
        self.loss_names = ['Task','CE']
        if self.isTrain and self.opt.validate:
            self.loss_names.append('validation')
        
        self.visual_names = ['rgb','color_gt', 'color_pred']  
        if self.isTrain and self.opt.validate:
            self.visual_names.append('validate_rgb')
            self.visual_names.append('color_validate_gt')
            self.visual_names.append('color_validate_pred')
        if not self.isTrain:
            self.visual_names = ['rgb','color_gt', 'color_pred', 'gt', 'mask']

        # network definition and initialization
        self.model_names = ['Task']
        self.netTask = networks.define_network(opt.input_nc, opt.output_nc, netTask=opt.netTask, gpu_ids=self.gpu_ids, pretrained=opt.pretrained)
        if self.isTrain:  # define discriminators
            self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_label)
            self.optimizer_G = torch.optim.Adam(self.netTask.parameters(), lr=opt.lr, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, data, isTrain=None):
        self.rgb = data['image'].to(self.device)
        self.gt = data['label'].to(self.device, dtype=torch.long)
        self.image_paths = data['name']

    def forward(self):
        # NOTE pertubate the inputs
        self.pred = self.netTask(self.rgb)
        self.mask = self.expand(self.get_hard_label(self.pred))

    def compute_visuals(self):
        with torch.no_grad():
            self.color_gt = self.expand(self.gt)
            self.color_pred = self.expand(self.get_hard_label(self.pred))
            self.gt = self.expand(self.gt)

    def backward_G(self):
        # Segmentation loss
        self.loss_CE = self.criterion_ce(self.pred, self.gt)
        # Overall loss
        self.loss_Task = self.loss_CE
        self.loss_Task.backward()

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()     
        self.forward()                  
        self.backward_G()                
        self.optimizer_G.step()          

    def validate(self, validation_data):
        # get input
        self.validate_rgb = validation_data['image'].to(self.device)
        self.validate_gt = validation_data['label'].to(self.device, dtype=torch.long)

        # forward
        self.validate_pred = self.netTask(self.validate_rgb)

        # get losses
        self.loss_validation = self.criterion_ce(self.validate_pred, self.validate_gt)


        # get visualizations
        self.color_validate_pred = torch.argmax(self.validate_pred, dim=1, keepdim=True)
        self.color_validate_gt = self.expand(self.validate_gt)

