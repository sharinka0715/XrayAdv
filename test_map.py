import os
import sys
import time
import argparse
import numpy as np
import cv2
import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from data import DATASET_ROOT, OPIXrayAnnotationTransform, OPIXrayDetection, IMAGE_SETS
from data import MODEL_CLASSES as labelmap
from ssd import build_ssd

warnings.filterwarnings("ignore")

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "a1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model', default="./ckpt/DOAM.pth", type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--dataset_root', default=DATASET_ROOT,
                    help='Location of dataset root directory')
parser.add_argument('--imagesetfile', default=IMAGE_SETS, type=str,
                    help='imageset file path to open')
parser.add_argument('--phase', default='test', type=str,
                    help='test phase')

args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        # print("WARNING: It looks like you have a CUDA device, but aren't using \
        #         CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    
    
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
    

def cal_ap(boxes, gts, npos, ovthresh=0.5):
    full_boxes = []
    for elm in boxes:
        if len(elm) > 0:
            full_boxes.append(elm)
    if len(full_boxes) == 0:
        return 0.0
    boxes = np.concatenate(full_boxes, 0) # [num_images*num_boxes, 5]
    
    # sort by confidence
    confidence = boxes[:, 4]
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    image_ids = [int(boxes[x, 5]) for x in sorted_ind]
    BB = boxes[sorted_ind, :]
    
    # mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    
    for d in range(nd):
        gt = np.array(gts[image_ids[d]])
        if len(gt) == 0:
            fp[d] = 1.
            continue
        bb = BB[d, :4].astype(float)
        ovmax = -np.inf
        BBGT = gt[:, :4].astype(float)
        
        # compute overlaps
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)
        inters = iw * ih
        uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
               (BBGT[:, 2] - BBGT[:, 0]) *
               (BBGT[:, 3] - BBGT[:, 1]) - inters)
        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
        if ovmax > ovthresh:
            if gts[image_ids[d]][jmax][5] == 0:
                tp[d] = 1.
                gts[image_ids[d]][jmax][5] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    
    rec = tp / np.maximum(float(npos), np.finfo(np.float64).eps)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    
    ap = voc_ap(rec, prec, False)
    return ap
        
    

def cal_map(all_boxes, all_gts):
    """
    calculate mAP as PASCAL VOC 2010
    params:
        all_boxes: 
            needs shape as (num_classes, num_images, num_boxes, 6)
            6 means [x1, y1, x2, y2, conf, img_id]
        all_gts:
            needs shape as (num_classes, num_images, num_boxes, 6)
            6 means [x1, y1, x2, y2, label, is_chosen(default to be 0)]
    """
    mAP = 0
    total = 0
    for i, cls in enumerate(labelmap):
        npos = 0
        for elm in all_gts[i]:
            npos += len(elm)
        if npos == 0:
            continue
        ap = cal_ap(all_boxes[i], all_gts[i], npos)
        print("AP for {}: {:.4f}".format(cls, ap))
        if not np.isnan(ap):
            mAP += ap
            total += 1
    print("mAP: {:.4f}".format(mAP / total))
    


def test_net(net, cuda, dataset, im_size=300):
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap))]
    
    all_gts = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap))]
    
    for i in tqdm(range(num_images)):
        im, gt, h, w, og_im, img_id = dataset.pull_item(i)
        im = im.type(torch.FloatTensor)
        # gt: [(x1, x2, y1, y2, label) * n], append 0 as chosen index
        for elm in gt:
            elm.append(0)
            all_gts[elm[4]][i].append(elm)
        
        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
            
        detections = net(x).data
        
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]

            scores = dets[:, 0].cpu().numpy()
            labels = np.ones_like(scores) * i
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis], labels[:, np.newaxis])).astype(np.float32, copy=False)
            all_boxes[j-1][i] = cls_dets
        
    cal_map(all_boxes, all_gts)


if __name__ == '__main__':
    num_classes = len(labelmap) + 1  # +1 for background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    
    dataset = OPIXrayDetection(args.dataset_root, args.imagesetfile,
                                  OPIXrayAnnotationTransform(), phase=args.phase)

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
        
    test_net(net, args.cuda, dataset, 300)
    print(args.trained_model, args.phase)

