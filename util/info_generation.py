import glob
import csv
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