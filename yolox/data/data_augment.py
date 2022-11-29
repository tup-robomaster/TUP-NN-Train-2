#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np

from shapely.geometry import Polygon,MultiPoint, polygon
from shapely.geometry.geo import box

from yolox.utils import xyxy2cxcywh

def augment_hsv(img, hgain=0.015, sgain=0.2, vgain=0.2):
    """
    HSV Data Augment
    """
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (gray.mean() < 20):
        r = np.random.uniform(0, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

def augment_noise(img,noise_density=0.03):
    """
    Pepper Noise Data Augment
    """
    half_density = noise_density / 2
    mask = np.random.uniform(0,1,img.shape[:2])
    img[mask < half_density] = [0,0,0]
    img[mask > (1 - half_density)] = [255,255,255]# no return needed
    
    
def is_outrange(img, box, padding_ratio=1, iou_thres=0.35, min_area=20):
    area_map = []
    for bbox in box:
        x = bbox[0::2]
        y = bbox[1::2]
        x_max = np.max(x)
        x_min = np.min(x)
        y_max = np.max(y)
        y_min = np.min(y)
        target_poly = Polygon(bbox.reshape(-1, 2))
        area = target_poly.area
        #If Max or min is out of range or label is out of range:
        if (x_max > img.shape[1] or x_min < 0 
            or y_max > img.shape[0] or y_min < 0 or area < min_area):
                img_pts = np.array([[0,0],[0, img.shape[0]],[img.shape[1],img.shape[0]],[img.shape[1], 0]])
                img_poly = Polygon(img_pts)
                intersect_area = target_poly.convex_hull.intersection(img_poly.convex_hull).area
                iou = intersect_area / area
                if iou > iou_thres and area >= min_area:
                    # print(iou)
                    area_map.append(True)
                    continue
                else:
                    center = np.array(target_poly.centroid.coords)
                    poly_vector = bbox.reshape(-1, 2) - np.repeat(center,len(bbox) / 2,axis=0)
                    # print("size:{}".format(area))
                    # print("{} : {}".format(poly_vector,poly_vector * padding_ratio))
                    padded_poly = np.array(bbox.reshape(-1, 2) + poly_vector * padding_ratio, dtype=np.int64)
                    cv2.fillConvexPoly(img, padded_poly, (0, 0, 0))
                    area_map.append(False)
        else:
            area_map.append(True)
    # print("...")
    return area_map

def box_candidates(box1, box2, wh_thr=10, ar_thr=10, area_thr=0.1):
    area_map = []
    for bbox1, bbox2 in zip(box1, box2):
        bbox1 = bbox1.reshape(-1,2)
        bbox2 = bbox2.reshape(-1,2)
        bbox1 = Polygon(bbox1)
        bbox2 = Polygon(bbox2)

        bbox1_bound_w = bbox1.bounds[2] - bbox1.bounds[0]
        bbox2_bound_w = bbox2.bounds[2] - bbox2.bounds[0]

        bbox1_bound_h = bbox1.bounds[3] - bbox1.bounds[1]
        bbox2_bound_h = bbox2.bounds[3] - bbox2.bounds[1]

        if (bbox1_bound_w != 0 and bbox1_bound_h != 0
                and bbox1_bound_h != 0 and bbox2_bound_h != 0):
            bbox1_wh = bbox1_bound_w / bbox1_bound_h
            bbox2_wh = bbox2_bound_w / bbox2_bound_h

            bbox_wh_ratio = abs(bbox1_wh - bbox2_wh)
            #print(bbox_wh_ratio)
        else:
            #For illegal widht and height
            bbox_wh_ratio = 100

        area = 0
        if bbox2.area != 0 and bbox1.area != 0:
            area = bbox2.area / bbox1.area
        area_map.append((area < ar_thr)&
                         (area > area_thr)&
                         (bbox_wh_ratio < wh_thr)
                         )
    area_map = np.array(area_map)
    return area_map


def random_perspective(
    img,
    targets=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    i = is_outrange(img=img, box=targets[:, :-2])
    targets = targets[i]

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )
    # Transform label coordinates
    n = len(targets)
    if n:
        apexes = int((targets.shape[1] - 2) / 2)
        # warp points
        xy = np.ones((n * apexes, 3))
        xy[:, :2] = targets[:, :-2].reshape(
            n * apexes, 2
        )  # x1y1, x2y2, x3y3 ..., xnyn
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, apexes * 2)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, apexes * 2)

        # # # clip boxes
        # xy[:, ::2] = xy[:, ::2].clip(0, width)
        # xy[:, 1::2] = xy[:, 1::2].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, :-2], box2=xy)
        targets = targets[i]
        targets[:, :-2] = xy[i]
        j = is_outrange(img=img, box=targets[:, :-2])
        targets = targets[j]

    return img, targets


def _mirror(image, boxes, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc(img, input_size, swap=(2, 0, 1)):
    """
    Preprocess input images.

    Resize and transpose image format from (y,x,channels) to (channels,x,y)
    """
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114
    # Caclute resize matrix
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # Transpose to (channels,x,y)
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    def __init__(self, num_apexes,max_labels=50, flip_prob=0.5, hsv_prob=1.0, noise_prob=0.2):
        self.num_apexes = num_apexes
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.noise_prob = noise_prob

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :self.num_apexes * 2].copy()
        labels = targets[:, self.num_apexes * 2:].copy()
        if len(boxes) == 0:
            # targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            targets = np.zeros((self.max_labels, self.num_apexes * 2 + 2), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :self.num_apexes * 2]
        labels_o = targets_o[:, self.num_apexes * 2:]

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        if random.random() < self.noise_prob:
            augment_noise(image)
        # mirror image
        image_t, boxes = _mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        # boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        # mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        # boxes_t = boxes[mask_b]
        # labels_t = labels[mask_b]
        boxes_t = boxes
        labels_t = labels

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        # expand dimension from [id,gt] to [id, max_batch, gt]
        # labels_t = np.expand_dims(labels_t, 1)
        # print()
        targets_t = np.hstack((labels_t, boxes_t))
        # padded_labels = np.zeros((self.max_labels, 5))
        padded_labels = np.zeros((self.max_labels, self.num_apexes * 2 + 2))
        padded_labels[range(len(targets_t))[:self.max_labels]] = targets_t[:self.max_labels]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))
