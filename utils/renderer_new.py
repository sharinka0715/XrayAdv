"""
X-ray renderer module.
Need kornia==0.2.2
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import kornia
import cv2
from tqdm import tqdm
import time

MIN_EPS = 1e-6

PARAM_IRON = [
    [0, 0, 104.3061111111111], 
    [-199.26894460884833, -1.3169138497713286, 227.17803542009827], 
    [-21.894450101465132, 0.20336113292167177, 274.63740523563814]
]

def load_from_file(path, mean=0.5, std=1/2.4):
    """
    Load vertices and faces from an .obj file.
    coordinates will be normalized to N(0.5, 0.5)
    """
    with open(path, "r") as fp:
        xlist = fp.readlines()

    vertices = []
    faces = []

    for elm in xlist:
        if elm[0] == "v":
            vertices.append([float(elm.split(" ")[1]), float(elm.split(" ")[2]), float(elm.split(" ")[3])])
        elif elm[0] == "f":
            faces.append([int(elm.split(" ")[1]) - 1, int(elm.split(" ")[2]) - 1, int(elm.split(" ")[3]) - 1])

    vertices = torch.Tensor(vertices).type(torch.cuda.FloatTensor)
    faces = np.array(faces, dtype=np.int32)
    return (vertices * std) + mean, faces

def is_in_triangle(point, tri_points):
    """
    Judge whether the point is in the triangle
    """
    tp = tri_points

    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = torch.dot(v0.T, v0)
    dot01 = torch.dot(v0.T, v1)
    dot02 = torch.dot(v0.T, v2)
    dot11 = torch.dot(v1.T, v1)
    dot12 = torch.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 < MIN_EPS:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v <= 1) & (inverDeno != 0)

def are_in_triangles(points, tri_points):
    """
    Judge whether the points are in the triangles
    assume there are n points, m triangles
    points shape: (n, 2)
    tri_points shape: (m, 3, 2)
    """
    tp = tri_points
    n = points.shape[0]
    m = tp.shape[0]

    # vectors
    # shape: (m, 2)
    v0 = tp[:, 2, :] - tp[:, 0, :]
    v1 = tp[:, 1, :] - tp[:, 0, :]
    # shape: (n, m, 2)
    v2 = points.unsqueeze(1).repeat(1, m, 1) - tp[:, 0, :]

    # dot products
    # shape: (m, 2) =sum=> (m, 1)
    dot00 = torch.mul(v0, v0).sum(dim=1)
    dot01 = torch.mul(v0, v1).sum(dim=1)
    dot11 = torch.mul(v1, v1).sum(dim=1)
    # shape: (n, m, 2) =sum=> (n, m, 1)
    dot02 = torch.mul(v2, v0).sum(dim=2)
    dot12 = torch.mul(v2, v1).sum(dim=2)

    # barycentric coordinates
    # shape: (m, 1)
    inverDeno = dot00*dot11 - dot01*dot01
    zero = torch.zeros_like(inverDeno)
    inverDeno = torch.where(inverDeno < MIN_EPS, zero, 1 / inverDeno)

    # shape: (n, m, 1)
    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno
    
    w0 = 1 - u - v
    w1 = v
    w2 = u

    # check if point in triangle
    return (u >= -MIN_EPS) & (v >= -MIN_EPS) & (u + v <= 1+MIN_EPS) & (inverDeno != 0), w0, w1, w2

def get_point_weight(point, tri_points):
    tp = tri_points
    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = torch.dot(v0.T, v0)
    dot01 = torch.dot(v0.T, v1)
    dot02 = torch.dot(v0.T, v2)
    dot11 = torch.dot(v1.T, v1)
    dot12 = torch.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 < MIN_EPS:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    w0 = 1 - u - v
    w1 = v
    w2 = u

    return w0, w1, w2


def ball2depth(vertices, faces, h, w):
    """
    Save obj file as a depth image, z for depth and x,y for position
    a ball with coord in [0, 1]
    h, w: the output image height and width
    return: a depth image in shape [h, w]
    """
    vertices = torch.clamp(vertices, 0, 1)
    vs = vertices.clone()
    vs[:, 0] = vertices[:, 0] * w
    vs[:, 1] = vertices[:, 1] * h
    vertices = vs

    # for thickness image
    min_depth = torch.zeros((h, w)) + 9999
    max_depth = torch.zeros((h, w)) - 9999

    for i in range(faces.shape[0]):
        tri = faces[i, :]

        # get rectangular bounding box
        umin = torch.floor(torch.min(vertices[tri, 0]))
        umin = torch.clamp(umin, 0, h-1)
        umax = torch.ceil(torch.max(vertices[tri, 0]))
        umax = torch.clamp(umax, 0, h-1)
        
        vmin = torch.floor(torch.min(vertices[tri, 1]))
        vmin = torch.clamp(vmin, 0, w-1)
        vmax = torch.ceil(torch.max(vertices[tri, 1]))
        vmax = torch.clamp(vmax, 0, w-1)

        # get depth value for each pixel in the triangular
        for u in range(int(umin.item()), int(umax.item())+1):
            for v in range(int(vmin.item()), int(vmax.item())+1):
                if not is_in_triangle(torch.Tensor([u, v]), vertices[tri, :2]):
                    continue
                w0, w1, w2 = get_point_weight(torch.Tensor([u, v]), vertices[tri, :2])
                point_depth = w0 * vertices[tri[0], 2] + w1 * vertices[tri[1], 2] + w2 * vertices[tri[2], 2]

                if point_depth <= min_depth[v, u]:
                    min_depth[v, u] = point_depth
                if point_depth >= max_depth[v, u]:
                    max_depth[v, u] = point_depth

    image = torch.clamp(max_depth - min_depth, 0, 1)
    image = torch.clamp(image, 0, 1)
    
    return image

def ball2depth_new(vertices, faces, h, w):
    """
    Save obj file as a depth image, z for depth and x,y for position
    a ball with coord in [0, 1]
    h, w: the output image height and width
    return: a depth image in shape [h, w]
    """
    vertices = torch.clamp(vertices, 0, 1)
    vs = vertices.clone()
    vs[:, 0] = vertices[:, 0] * w
    vs[:, 1] = vertices[:, 1] * h
    vertices = vs
    faces = torch.LongTensor(faces).cuda()
    
    points = torch.Tensor([(i, j) for i in range(h) for j in range(w)]).cuda()
    tri_points = vertices[faces, :2]
    in_triangle, w0, w1, w2 = are_in_triangles(points, tri_points)
    
    point_depth = w0 * vertices[faces[:, 0], 2] + w1 * vertices[faces[:, 1], 2] + w2 * vertices[faces[:, 2], 2]
    
    min_depth = torch.min(torch.where(in_triangle, point_depth, torch.full_like(point_depth, 9999)), dim=1).values
    max_depth = torch.max(torch.where(in_triangle, point_depth, torch.full_like(point_depth, -9999)), dim=1).values

    image = torch.clamp(max_depth - min_depth, 0, 1).view(h, w)
    
    return image


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    for model in ["../objs/ball_small.obj", "../objs/ball_d.obj"]:
        for size in [20, 40]:
            vertices, faces = load_from_file(model)
            print("Object & Patch size: size={}, faces={}".format(size, faces.shape[0]))

            timer = time.time()
            image = ball2depth(vertices, faces, size, size)
            torch.cuda.synchronize()
            print("Before acceleration: {:.4f}s".format(time.time() - timer))
            cv2.imwrite("save_before.png", np.clip(image.cpu().numpy() * 255, 0, 255).astype(np.uint8))

            timer = time.time()
            image = ball2depth_new(vertices, faces, size, size)
            torch.cuda.synchronize()
            print("After acceleration: {:.4f}s\n".format(time.time() - timer))
            cv2.imwrite("save_after.png", np.clip(image.cpu().numpy() * 255, 0, 255).astype(np.uint8))