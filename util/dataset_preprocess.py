import glob
import csv
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from os import path as osp

def generate_info(root, dtype='.png'):
    """Iterate over the directory of dataset and generate image informatino for each sample
    
    Parameters:
        root                -- directory of the dataset
        dtype               -- type of image sample
    
    Returns:
        infos               -- information dictionary of each image
    """
    infos = []
    for path in glob.glob(osp.join(root, 'image', '*' + dtype)):
        name = path.split('/')[-1]
        info = {
            'image_path': path,
            'label_path': osp.join(root, 'label', name),
            'image_name': name
        }
        infos.append(info)
    return infos

def generate_class_map(root, name='mapping.csv'):
    """Class Mapping format is located under root directory of the dataset, with the format of "Class id, Class name, New id, New name"
    """
    mapping = {}
    with open(osp.join(root, name+'.csv'), 'r') as f:
        text = csv.reader(f, delimiter='\t')
        text = list(text)[1:]
        for item in text:
            [cid, cname, newid, newname] = item
            mapping[int(cid)] = {"name":cname, "newid":int(newid), "newname":newname}
    return mapping


def get_normalize_statistics(root, phases=['training', 'test'], dtype='.png'):
    """calculate mean and variance of values of pixels in dataset
    """
    transform = T.ToTensor()
    for phase in phases:
        statistics = torch.zeros((2, 3)) # mean, variance
        infos = generate_info(osp.join(root, phase), dtype)
        for info in infos:
            image = Image.open(info['image_path'])
            image = transform(image)
            image = image.view(3, -1)
            statistics[0] += torch.mean(image, dim=1)
            statistics[1] += torch.std(image, dim=1)
        statistics /= len(infos)
        print('length of the dataset:{}, mean:{}, variance:{}'.format(len(infos), statistics[0], statistics[1]))
        with open(osp.join(osp.join(root, phase, 'statistics.txt')), 'w') as f:
            mean = statistics[0]
            variance = statistics[1]
            mean = '{},{},{}\n'.format(mean[0], mean[1], mean[2])
            variance = '{},{},{}'.format(variance[0], variance[1], variance[2])
            f.write(mean)
            f.write(variance)

def get_normalize_statistics(root):
    with open(osp.join(root, 'statistics.txt'), 'r') as f:
        reader = csv.reader(f)
        values = list(reader)
        values = list(map(lambda x: [float(item) for item in x], values))
        values = torch.tensor(values)
    return (values[0], values[1])

if __name__ == '__main__':
    (mean, std) = get_normalize_statistics('/home/comglq/Documents/iMED/projects/Conference_2023_summer_code_bank/Conference2023/datasets/CaDIS_clean/training')
    transform
            
