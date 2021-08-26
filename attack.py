"""
Xray Adversarial Attack
Author: Jun Guo
"""
import os
import sys
import cv2
import math
import time
import yaml
import torch
import random
import kornia
import argparse
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from torch.autograd import Variable, gradcheck
from torch.utils.data import DataLoader
from easydict import EasyDict
from pprint import pprint

from data import OPIXrayDetection, OPIXrayAnnotationTransform, detection_collate, DATASET_ROOT, IMAGE_SETS
from ssd import build_ssd
from utils import stick, renderer
from layers import MultiBoxLoss

warnings.filterwarnings("ignore")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser = argparse.ArgumentParser(description="X-ray adversarial attack.")
# for model
parser.add_argument("--ckpt_path", default="./ckpt/DOAM.pth", type=str, 
                    help="the checkpoint path of the model")
# for data
parser.add_argument("--dataset_root", default=DATASET_ROOT, type=str, 
                    help="the root of the X-ray image dataset")
parser.add_argument("--imagesetfile", default=IMAGE_SETS, type=str, 
                    help="the image sets of the X-ray image dataset")
parser.add_argument("--phase", default="test", type=str, 
                    help="the phase of the X-ray image dataset")
parser.add_argument("--batch_size", default=10, type=int, 
                    help="the batch size of the data loader")
parser.add_argument("--num_workers", default=4, type=int, 
                    help="the number of workers of the data loader")
# for patch
parser.add_argument("--obj_path", default="objs/ball_small.obj", type=str, 
                    help="the path of adversarial 3d object file")
parser.add_argument("--patch_size", default=20, type=int, 
                    help="the size of X-ray patch")
parser.add_argument("--patch_count", default=4, type=int, 
                    help="the number of X-ray patch")
parser.add_argument("--patch_place", default="search", type=str, choices=['fix', 'search', 'backfire'],
                    help="the place where the X-ray patch located")
# for attack
parser.add_argument("--lr", default=0.01, type=float, 
                    help="the learning rate of attack")
parser.add_argument("--beta", default=10, type=float, 
                    help="the learning rate of attack")
parser.add_argument("--num_iters", default=24, type=int, 
                    help="the number of iterations of attack")
parser.add_argument("--save_path", default=None, type=str,
                    help="the save path of adversarial examples")


timer = time.time()
def stime(content):
    global timer
    torch.cuda.synchronize()
    print(content, time.time() - timer)
    timer = time.time()


def set_seed(seed=0):
    """
    Fix the random seed to make the experiments reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def get_place_fix(images, targets, group, faces, net, criterion):
    fix_place_list = ["nw", "ne", "sw", "se", "n", "s", "w", "e"]
    areas_choose = [[] for _ in range(images.shape[0])]
    for i in range(args.patch_count):
        places = stick.cal_stick_place(stick.parse_gtbox(targets), args.patch_size, args.patch_size, 0.25, fix_place_list[i])
        for j in range(len(places)):
            areas_choose[j].append(places[j])
            
    return areas_choose
    
    
def get_place_search(images, targets, group, faces, net, criterion):
    """
    Calculate the best stick place for patches.
    """
    pad = nn.ZeroPad2d(args.patch_size)
    group_clamp = torch.clamp(group, 0, 1)
    
    # use X-ray renderer to convert a 3D object to an X-ray image
    rend_group = []
    for pt in range(args.patch_count):
        depth_img = renderer.ball2depth(group_clamp[pt], faces, args.patch_size, args.patch_size).unsqueeze(0).unsqueeze(0)
        # simulate function needs a 4-dimension input
        rend, mask = renderer.simulate(depth_img)
        rend[~mask] = 1
        rend_group.append(rend)
    
    areas = stick.get_stick_area(stick.parse_gtbox(targets), args.patch_size, args.patch_size)
    areas_choose = []
    images_c = []
    
    for bi in range(images.shape[0]):
        print("Image {} in batch size {}...".format(bi+1, images.shape[0]))
        area = areas[bi]
        area_choose = []
        image_c = pad(images[bi:bi+1].clone().detach())
        
        for pt in range(args.patch_count):
            time_now = time.time()
            rend = rend_group[pt]
            if len(area) == 0:
                print("No area to paste!")
                area_choose.append(area_choose[0])
                continue
            with torch.no_grad():
                loss_c_list = []
                images_forward = []
                for (u, v) in area:
                    images_delta = image_c.clone().detach()
                    images_delta[:, :, u+args.patch_size:u+2*args.patch_size, v+args.patch_size:v+2*args.patch_size].mul_(rend)
                    images_forward.append(images_delta)
                images_forward = torch.cat(images_forward, dim=0)
                # forward every 50 images
                outs = [[], [], None]
                for ifor in range((images_forward.shape[0] + 49) // 50):
                    out = net(images_forward[ifor*50:ifor*50+50, :, 
                                             args.patch_size:300+args.patch_size, args.patch_size:300+args.patch_size])
                    outs[0].append(out[0])
                    outs[1].append(out[1])
                    outs[2] = out[2]
                outs[0] = torch.cat(outs[0], dim=0)
                outs[1] = torch.cat(outs[1], dim=0)
                for iout in range(images_forward.shape[0]):
                    out = (outs[0][iout:iout+1], outs[1][iout:iout+1], outs[2])
                    _, loss_c = criterion(out, [targets[bi]])
                    loss_c_list.append(loss_c.item())
                loss_c_list = np.array(loss_c_list)
                index = loss_c_list.argmax()
                area_choose.append(area[index])
                u, v = area[index]
                image_c[:, :, u+args.patch_size:u+2*args.patch_size, v+args.patch_size:v+2*args.patch_size].mul_(rend)

                # delete overlap place
                new_area = []
                for (tu, tv) in area:
                    if abs(u - tu) < args.patch_size / 2 and abs(v - tv) < args.patch_size / 2:
                        continue
                    new_area.append((tu, tv))
                area = new_area

                torch.cuda.synchronize()
                print("Find area coord {}, loss={:.2f} Time: {:.4f}".format(
                    area_choose[-1], loss_c_list[index].item()*1000, time.time() - time_now))
                torch.cuda.empty_cache()
                
        areas_choose.append(area_choose)
             
        images_c.append(image_c)
        
    return areas_choose, torch.cat(images_c, dim=0)
    
    
def attack(images, targets, net, criterion):
    """
    Main attack function.
    """
    net.phase = "train"
    images = images.type(torch.cuda.FloatTensor)
    targets = [Variable(ann.cuda(), requires_grad=False) for ann in targets]
    
    # create a group of patch objects which have same faces
    # we only optimize the coordinate of vertices
    # but not to change the adjacent relation
    group = []
    for _ in range(args.patch_count):
        vertices, faces = renderer.load_from_file(args.obj_path)
        group.append(vertices.unsqueeze(0))

    adj_ls = renderer.adj_list(vertices, faces)
    
    # the shape of group: [patch_count, 3, vertices_count]
    group = torch.cat(group, dim=0).cuda()
    group.requires_grad_(True)
    group_ori = group.clone().detach()
    
    # we need a pad function to prevent that a part of patch is out of the image
    optimizer = optim.Adam([group], lr=args.lr)
    pad = nn.ZeroPad2d(args.patch_size)
    
    print("Calculate best place before attack...")
    if args.patch_place == "search":
        areas_choose, _ = get_place_search(images, targets, group, faces, net, criterion)
    elif args.patch_place == "fix":
        areas_choose = get_place_fix(images, targets, group, faces, net, criterion)
    
    print("Attacking...")
    for t in range(args.num_iters):
        timer = time.time()
        
        images_delta = images.clone().detach()
        images_delta = pad(images_delta)
        
        # calculate the perspective loss
        loss_per = torch.Tensor([0])
        for pt in range(args.patch_count):
            loss_per += renderer.tvloss(group_ori[pt], group[pt], adj_ls, coe=0)
        loss_per /= args.patch_count
        
        # clamp the group into [0, 1]
        group_clamp = torch.clamp(group, 0, 1)
        
        # use X-ray renderer to convert a 3D object to an X-ray image
        for pt in range(args.patch_count):
            depth_img = renderer.ball2depth(group_clamp[pt], faces, args.patch_size, args.patch_size).unsqueeze(0).unsqueeze(0)
            # simulate function needs a 4-dimension input
            rend, mask = renderer.simulate(depth_img)
            rend[~mask] = 1
            for s in range(images_delta.shape[0]):
                u, v = areas_choose[s][pt]
                images_delta[s:s+1, :, u+args.patch_size:u+2*args.patch_size, v+args.patch_size:v+2*args.patch_size].mul_(rend)
#         with torch.autograd.detect_anomaly():
#             net.phase = "test"
        out = net(images_delta[:, :, args.patch_size:300+args.patch_size, args.patch_size:300+args.patch_size])
#             net.phase = "train"
#         pred = nn.Softmax(dim=-1)(out[1])[:, :, 0]
#         loss_adv = - torch.mean(torch.log(pred[pred>0]))
        
        _, loss = criterion(out, targets)
        loss_adv = - loss

#         pred = out[:, :, :, 0]
#         loss_adv = -torch.mean(torch.log(pred[pred>0]))
        loss_total = loss_adv + args.beta * loss_per

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        
        print("Iter: {}/{}, L_adv = {:.3f}, βL_per = {:.3f}, Total loss = {:.3f}, Time: {:.2f}".format(
            t+1, args.num_iters, loss_adv.item() * 1000, args.beta * loss_per.item() * 1000,
            loss_total.item() * 1000, time.time() - timer))
            
    print("Calculate best place after attack...")
    if args.patch_place == "search":
        areas_choose, images_adv = get_place_search(images, targets, group, faces, net, criterion)
    elif args.patch_place == "fix":
        areas_choose = get_place_fix(images, targets, group, faces, net, criterion)
        group_clamp = torch.clamp(group, 0, 1)
        images_adv = pad(images.clone().detach())
        for pt in range(args.patch_count):
            depth_img = renderer.ball2depth(group_clamp[pt], faces, args.patch_size, args.patch_size).unsqueeze(0).unsqueeze(0)
            # simulate function needs a 4-dimension input
            rend, mask = renderer.simulate(depth_img)
            rend[~mask] = 1
            for s in range(images_delta.shape[0]):
                u, v = areas_choose[s][pt]
                images_adv[s:s+1, :, u+args.patch_size:u+2*args.patch_size, v+args.patch_size:v+2*args.patch_size].mul_(rend)
        
    return images_adv[:, :, args.patch_size:300+args.patch_size, args.patch_size:300+args.patch_size], areas_choose, torch.clamp(group, 0, 1), faces
    
def save_img(path, img_tensor):
    img_tensor = img_tensor.cpu().detach().numpy().astype(np.uint8)
    img = img_tensor.transpose(1, 2, 0)
    img = cv2.resize(img, (1225, 954))
    cv2.imwrite(path, img)
    
if __name__ == "__main__":
    set_seed(17373051)

    args = parser.parse_args()
    
    net = build_ssd("test", size=300, num_classes=6)
    net.load_weights(args.ckpt_path)

    print("CUDA is available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("Warning! CUDA is not supported on your device!")
        sys.exit(0)
    else:
        print("CUDA visible device count:", torch.cuda.device_count())
        
    cudnn.benchmark = True
    net = net.cuda()
    net.eval()

    dataset = OPIXrayDetection(root=args.dataset_root, image_sets=args.imagesetfile, phase='test')
    data_loader = DataLoader(dataset, args.batch_size, shuffle=True, collate_fn=detection_collate, pin_memory=True)

    criterion = MultiBoxLoss(6, 0.5, True, 0, True, 3, 0.5, False, True)
    num_images = len(dataset)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    img_path = os.path.join(args.save_path, "adver_image")
    if not os.path.exists(img_path):
        os.makedirs(img_path)
        
    obj_path = os.path.join(args.save_path, "adver_obj")
    if not os.path.exists(obj_path):
        os.makedirs(obj_path)

    for i, (images, targets, img_ids) in enumerate(data_loader):
        print("Batch {}/{}...".format(i+1, math.ceil(num_images / args.batch_size)))
        images_adv, areas_choose, vertices, faces = attack(images, targets, net, criterion)
        
        print("Saving...")
        for t in range(images_adv.shape[0]):
            save_img(os.path.join(img_path, img_ids[t] + ".png"), images_adv[t])
            if faces is not None:
                for i in range(vertices.shape[0]):
                    renderer.save_to_file(
                        os.path.join(obj_path, str(img_ids[t]) + "_u{}_v{}.obj".format(*areas_choose[t][i])), 
                        vertices[i], faces)
