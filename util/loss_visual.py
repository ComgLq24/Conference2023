import visdom
import numpy as np
import argparse
import os.path as osp
import sys
import re
from subprocess import Popen, PIPE

class LossVisualizer(object):
    """
    使用前必须保证loss_log.txt里面只有一次训练的损失
    """
    def __init__(self, opt):
        self.opt = opt
        self.path = osp.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        self.vis = visdom.Visdom(env='loss_visualization', port=8099)
        if not self.vis.check_connection():
            self.create_visdom_connections()

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p 8087 &>/dev/null &'
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    
    def visualize_losses(self):
        for key, values in self.losses.items():
            # print(values)
            self.vis.line(values, list(range(1, len(values)+1)), win=key, opts=dict(title=key))

    def get_losses(self):
        losses = {}
        with open(self.path, 'r') as f:
            lines = f.readlines()[1:]
            for x, line in enumerate(lines):
                line = line.split(')')[1].strip()
                values = re.findall('[\d]+.?[\d]*', line)
                names = re.findall('[a-zA-Z]+', line)
                for index in range(len(values)):
                    if not names[index] in losses.keys():
                        losses[names[index]] = [float(values[index])]
                    else:
                        losses[names[index]].append(float(values[index]))
        self.losses = losses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_dir', default='./checkpoints', type=str)
    parser.add_argument('--name', type=str, required=True)
    opt = parser.parse_args()
    visualizer = LossVisualizer(opt)
    visualizer.get_losses()
    visualizer.visualize_losses()