"""
OPIXray Dataset Classes
"""
import os
#from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

MODEL_CLASSES = (
    'Folding_Knife', 'Straight_Knife', 'Scissor', 'Utility_Knife', 'Multi-tool_Knife',
)

DATASET_ROOT = "/home/jovyan/17373051/XRayAdversary/OPIXray/Dataset/test/"
IMAGE_SETS = "/home/jovyan/17373051/XRayAdversary/OPIXray/Dataset/test/test_knife.txt"

class OPIXrayAnnotationTransform(object):
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

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(MODEL_CLASSES, range(len(MODEL_CLASSES))))
        self.keep_difficult = keep_difficult
        self.type_dict = {}
        self.type_sum_dict = {}
    def __call__(self, target, width, height, idx):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
            it has been changed to the path of annotation-2019-07-10
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        #print (idx)
        res = []
        with open(target, "r", encoding='utf-8') as f1:
            dataread = f1.readlines()
        for annotation in dataread:
            bndbox = []
            temp = annotation.split()
            name = temp[1]

            if name not in MODEL_CLASSES:
                continue
            xmin = int(temp[2]) / width
            if xmin > 1:
                continue
            if xmin < 0:
                xmin = 0
            ymin = int(temp[3]) / height
            if ymin < 0:
                ymin = 0
            xmax = int(temp[4]) / width
            if xmax > 1:           
                xmax = 1
            ymax = int(temp[5]) / height
            if ymax > 1:
                ymax = 1
            bndbox.append(xmin)
            bndbox.append(ymin)
            bndbox.append(xmax)
            bndbox.append(ymax)

            label_idx = self.class_to_ind[name]
            # label_idx = name
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        if len(res) == 0:
            return [[0, 0, 0, 0, 5]]
        return res

def test_Sobel(img):
    #src = cv2.imread(src)
    #gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst

class OPIXrayDetection(data.Dataset):
    def __init__(self, root=DATASET_ROOT,
                 image_sets=IMAGE_SETS,
                 transform=None, target_transform=OPIXrayAnnotationTransform(),phase=None):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        #self.name = dataset_name
        self.name = 'OPIXray_knife'
        if(phase == 'test'):
            self._annopath = osp.join('%s' % self.root, 'test_annotation', '%s.txt')
            self._imgpath = osp.join('%s' % self.root, 'test_image', '%s.TIFF')
            self._imgpath1 = osp.join('%s' % self.root, 'test_image', '%s.tiff')
            self._imgpath2 = osp.join('%s' % self.root, 'test_image', '%s.jpg')
        elif(phase == 'train'):
            self._annopath = osp.join('%s' % self.root, 'train_annotation', '%s.txt')
            self._imgpath = osp.join('%s' % self.root, 'train_image', '%s.TIFF')
            self._imgpath1 = osp.join('%s' % self.root, 'train_image', '%s.tiff')
            self._imgpath2 = osp.join('%s' % self.root, 'train_image', '%s.jpg')
        else:
            print('No phase')
            self._annopath = osp.join('%s' % self.root, 'test_annotation', '%s.txt')
            self._imgpath = osp.join('%s' % self.root, phase, '%s.TIFF')
            self._imgpath1 = osp.join('%s' % self.root, phase, '%s.png')
            self._imgpath2 = osp.join('%s' % self.root, phase, '%s.jpg')
        self.ids = list()

        #listdir = os.listdir(osp.join('%s' % self.root, 'Annotation'))

        with open(self.image_set, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.ids.append(line.strip('\n'))
            

    def __getitem__(self, index):
        im, gt, h, w, og_im, img_id = self.pull_item(index)

        return im, gt, img_id

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        #target = ET.parse(self._annopath % img_id).getroot()

        target = self._annopath % img_id
        #print(target)
        #print(self._imgpath % img_id)
        img = cv2.imread(self._imgpath % img_id)
        if img is None:
            img = cv2.imread(self._imgpath1 % img_id)
        if img is None:
            img = cv2.imread(self._imgpath2 % img_id)
        if img is None:
            print('wrong')
        #print()
        #print(self._imgpath2 % img_id)
        try:
            height, width, channels = img.shape
        except:
            print(img_id)
        #print("height: " + str(height) + " ; width : " + str(width) + " ; channels " + str(channels) )
        og_img = img
        #yuv_img = cv2.cvtColor(og_img,cv2.COLOR_BGR2YUV)
        try:
            img = cv2.resize(img,(300,300))
        except:
            print('img_read_error')

        #img = np.concatenate((img,sobel_img),2)
        #print (img_id)
        if self.target_transform is not None:
            target = self.target_transform(target, width, height, img_id)
            
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width, og_img, img_id