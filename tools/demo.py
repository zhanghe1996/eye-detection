#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

COUNT = 0
CONF_THRESH = 0.80
NMS_THRESH = 0.3

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--class', dest='my_class', help='Network to use [vgg16]',
                        default='__background__', type=str)

    args = parser.parse_args()

    return args

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    args = parse_args()
    #image_path = os.path.join(cfg.ROOT_DIR, 'data', 'INRIA_Pupil_devkit',
    #            args.my_class, 'data', 'Images', image_name + '.jpeg')
    image_path = os.path.join(cfg.ROOT_DIR, 'data', 'INRIA_Pupil_devkit',
                args.my_class + '_test', 'data', 'Images', image_name)

    # Load the demo image
    im = cv2.imread(image_path)

    # Detect all object classes and regress object bounds
    #timer = Timer()
    #timer.tic()
    scores, boxes = im_detect(net, im)
    
    #timer.toc()
    #print ('Detection took {:.3f}s for '
    #       '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    
    print scores
    sorted_scores = sorted(scores[:, 1], reverse = True)
    print sorted_scores[0], sorted_scores[1]
    if sorted_scores[1] >= CONF_THRESH:
        global COUNT
        COUNT += 1
    
    cls_ind = 1
    cls_boxes = boxes[:, 4 * cls_ind : 4 * (cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    
    keep = np.where(cls_scores >= CONF_THRESH)[0]
    cls_boxes = cls_boxes[keep, :]
    cls_scores = cls_scores[keep]
    
    cls_scores = np.sort(cls_scores)
    
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    print dets

    output_path = os.path.join(cfg.ROOT_DIR, 'data', 'INRIA_Pupil_devkit',
                    args.my_class + '_test', 'data', 'Annotations', image_name.split('.')[0] 
                    + '_' + args.my_class + '.npy')
    f = open(output_path, 'w+')
    np.save(f, dets)
    f.close()

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    
    args = parse_args()
    prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'faster_rcnn_alt_opt', 
                'faster_rcnn_test.pt')
    #prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'pascal_voc', 'VGG_CNN_M_1024',
    #            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    
    caffemodel = os.path.join(cfg.ROOT_DIR, 'models', args.my_class,
                args.my_class + '_faster_rcnn_final.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    file_path = os.path.join(cfg.ROOT_DIR, 'data', 'INRIA_Pupil_devkit',
                args.my_class + '_test', 'data', 'ImageSets', 'test.txt')
    #file_path = os.path.join(cfg.ROOT_DIR, 'data', 'INRIA_Pupil_devkit',
    #            args.my_class, 'data', 'ImageSets', 'test.txt')
    f = open(file_path, 'r')
    image_names = f.read().splitlines()
    f.close()
    #print video_names
    out_path = os.path.join(cfg.ROOT_DIR, 'data', 'scores', args.my_class)
    
    for image_name in image_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Test on image: {}'.format(image_name)
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

        demo(net, image_name)
        
   
    print 'COUNT: {}'.format(COUNT)

