import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

CLASSES = ( '__background__', 'face')

class AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self):
        self.class_to_ind = dict( zip(CLASSES, range(len(CLASSES))))
        #self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))
        #for obj in target.iter('object'):
            #difficult = int(obj.find('difficult').text) == 1
            #if not self.keep_difficult and difficult:
            #    continue
            #name = obj.find('name').text.lower().strip()
            #bbox = obj.find('bndbox')
            #pts = ['xmin', 'ymin', 'xmax', 'ymax']
            #bndbox = []
            #for i, pt in enumerate(pts):
            #    cur_pt = int(bbox.find(pt).text)
            #    bndbox.append(cur_pt)
            #label_idx = self.class_to_ind[name]
            #bndbox.append(label_idx)
            #res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
        label_idx = self.class_to_ind['face']
        bndbox = [int(target[0]),int(target[1]),int(target[2]),int(target[3])]
        bndbox.append(label_idx)
        res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
        return res


class AnimeDetection(data.Dataset):

    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to WIDER folder
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
    """

    def __init__(self, filelist, preproc=None, target_transform=None):
        self.filelist = filelist
        self.preproc = preproc
        self.target_transform = target_transform
        #self._annopath = os.path.join(self.root, 'annotations', '%s')
        #self._imgpath = os.path.join(self.root, 'images', '%s')
        self.ids = list()
        with open(self.filelist, 'r') as f:
          self.ids = [tuple(line.split()) for line in f]

    def __getitem__(self, index):
        img_id = self.ids[index]
        #print(img_id)
        #target = ET.parse(self._annopath % img_id[1]).getroot()
        target = img_id[1:]
        img = cv2.imread(img_id[0], cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

    def __len__(self):
        return len(self.ids)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
