import os.path as osp
import csv
import numpy as np
import argparse
from PIL import Image

# TODO 写IoU metrics
# TODO 测试的时候添加类别信息
# TODO 测试的时候添加namelist
# TODO 写Dice

class SegMetrics(object):
    
    def __init__(self, opt):
        self.opt = opt
        self.results_dir = osp.join(opt.results_dir, 'test_'+opt.epoch)
        self.images_dir = osp.join(self.results_dir, 'images')
        self.classes = self.get_classes(osp.join(self.results_dir, 'classes.csv'))
        self.num_classes = len(self.classes)
        self.names = self.list_names(osp.join(self.results_dir, 'names.txt'))
        # self.metrics = opt.metrics.split(',')

    def list_names(self, names_path):
        name_list = []
        with open(names_path, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                name_list.append(line)
        return name_list

    def get_classes(self, classes_path):
        classes = {}
        with open(classes_path, 'r') as f:
            lines = list(csv.reader(f,delimiter=','))[1:]
            for line in lines:
                index = int(line[0])
                name = line[1]
                classes[index] = {'class_name':name, 'gt':0, 'mask':0}
        return classes
    
    def cal_statistics(self, gt, mask):
        return  np.bincount(gt*self.num_classes + mask, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def cal_metrics(self):
        matrix = np.zeros((self.num_classes, self.num_classes), np.float64)
        for name in self.names:
            gt = Image.open(osp.join(self.images_dir, name+'_gt.png'))
            gt = np.asarray(gt, dtype=np.uint32)
            mask = Image.open(osp.join(self.images_dir, name+'_mask.png'))
            mask = np.asarray(mask, dtype=np.uint32)

            statistics = self.cal_statistics(gt.flatten(), mask.flatten())
            matrix = matrix + statistics
        self.matrix = matrix
        self.gts = np.sum(matrix, axis=0)
        self.masks = np.sum(matrix, axis=1)
        for index in self.classes:
            self.classes[index]['gt'] = self.gts[index]
            self.classes[index]['mask'] = self.masks[index]

    def iou(self):
        intersections = np.diag(self.matrix)
        unions = np.sum(self.matrix, axis=0) + np.sum(self.matrix, axis=1) - intersections
        ious = np.divide(intersections, unions)
        for index in self.classes.keys():
            self.classes[index]['iou'] = ious[index]

    def print_results(self):
        record=""
        with open(osp.join(self.results_dir, 'metrics.txt'), 'w') as fi:
            for results in self.classes.values():
                for key, value in results.items():
                    if type(value) == str:
                        record += "%s:\t\t"%value
                    else:
                        record +="%s %.04f   " % (key, value*100)
                record+="\n"
                # results = list(results.items())
                # print(results)
            fi.write(record)
        print(record, end="")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--epoch', type=str, required=True)
    metrics = SegMetrics(parser.parse_args())
    metrics.cal_metrics()
    metrics.iou()
    metrics.print_results()
            
            