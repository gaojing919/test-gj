# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
import sys
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math

def set_scale(boxes):
    b_scale = np.zeros(40)
    #boxes = np.array(boxes[:,0:4])
    b_large_size = np.zeros(len(boxes))
    boxes = np.array(boxes[:,0:4])
    #print(boxes)
    #xmin = np.array(boxes[0][:])
    #ymin = boxes[:][1]
    #xmax = boxes[:][2]
    #ymax = boxes[:][3]
    #w = np.array(boxes[:,2]-boxes[:,0])
    #h = np.array(boxes[:,3]-boxes[:,1])
    size = np.array([boxes[:,2]-boxes[:,0],boxes[:,3]-boxes[:,1]])
    #print(w)
    #print(h)
    #print("gjsize")
    #print(size)
    #尺度定义为面积开根号，除以16
    for i in range(size.shape[1]):
        b_large_size[i] = math.sqrt(size[0,i]*size[1,i])/16
    #print(b_large_size)
    model_count = 0
    model_index = []
    for b_size in b_large_size:
        #i = len(bin(int(b_size)))-4
        #index = i*10-1
        #if(index<0):
        #    index = 9
        #if(index>59):
        #    index = 59
        '''
        x = len(bin(int(b_size)))-2
        if(abs(b_size-2**x)<abs(b_size-2**(x-1))):
          y = x
        else:
          y = x-1
        #print(b_size)
        #print()

        index = (y-2)*10+5
        '''
        index = int(round(b_size)-1)

        #print(index)
        #i = len(bin(int(b_size)))-4
        #index = i*10-1
        if(index<0):
            #index = 5
            index = 0
        if(index>39):
            index = 39
        '''
        if(index>35):
            index = 35
        '''
        if(b_scale[index]==0):
            b_scale[index]=1
            model_count = model_count+1
            model_index.append(index)
            #print(index)
    #print(b_large_size)
    #print("gj_yuan")
    #print(model_index)
    #for o in model_index:
    #    print(2**((o/10)-5)+8)
    for i in range(b_scale.size):
        d = 0
        for x in model_index:
            if(abs(i-x)<20):
                d = d+1
        for j in model_index:
            if(abs(i-j)<20):
                norm = multivariate_normal(mean=j, cov=1/math.sqrt(2*math.pi))
                if(d!=0):
                    r = 1/d
                else:
                    r = 0
                b_scale[i] = b_scale[i]+r*norm.pdf(i)
                if(math.isnan(b_scale[i])):
                    print("nan")
                    print(d)
                    print(norm.pdf(i))
                if(b_scale[i]>1):
                    b_scale[i]=1
                    #print("big1")
    #plt.plot(range(60),b_scale)
    #plt.show()
        #print(i)
        #print(len(i)-2)
    #b_large_size[:] = bin(b_large_size)
    #for b in boxes:
    #print("gjb_scale")
    #print(b_large_size)

    #print("gj_boxes")
    #print(boxes)
    #print(model_index)
    #b_scale.reshape
    #sys.exit()
    #for x in range(b_scale.size):
       # b_scale[x] = [[b_scale[x]]]
    #b_scale.reshape(60,1,1)
    return b_scale


def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    if cfg.TRAIN.HAS_RPN:
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # gt boxes: (x1, y1, x2, y2, cls)
        #print("gjroidb")
        #print(roidb)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        #print("gjgt_boxes[:, 4]")
        #print(gt_boxes[:, 4])
        blobs['gt_boxes'] = gt_boxes
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)
        
        #print("roidb[0]['boxes'][gt_inds, :]")
        #print(roidb[0]['boxes'][gt_inds, :])
        #blobs['gt_scale'] = set_scale(roidb[0]['boxes'][gt_inds, :])
        blobs['gt_scale'] = set_scale(gt_boxes[:, 0:4].reshape(len(gt_boxes),4))
        #print("gjdb_inds")
        #print(blobs['gt_scale'])
        #print(blobs['gt_boxes'])
    else: # not using RPN
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
        #scale_blob = np.zeros((0, 40), dtype=np.float32)
        # all_overlaps = []
        for im_i in range(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights \
                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                               num_classes)

            # Add to RoIs blob
            #print("gjtarge")
            #print(bbox_targets.shape)
            #print(bbox_targets)
            rois = _project_im_rois(im_rois, im_scales[im_i])
            batch_ind = im_i * np.ones((rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, rois))
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))
            gt_inds = np.where(roidb[im_i]['gt_classes'] != 0)[0]
            #bb = roidb[im_i]['boxes'][gt_inds, :] * im_scales[im_i]
            #print(bb.shape)
            #print(im_i)
            #scales = set_scale(bb.reshape(len(bb),4))

            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob = np.hstack((labels_blob, labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
           
            #print(bbox_targets_blob.shape)
            #scale_blob = np.vstack((scale_blob, scales))
            # all_overlaps = np.hstack((all_overlaps, overlaps))

        # For debug visualizations
        # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)
        #print("gjbbox_targets_blob")

        blobs['rois'] = rois_blob
        blobs['labels'] = labels_blob
        #blobs['gt_scale'] = scale_blob
        #print(blobs['gt_scale'].shape)

        if cfg.TRAIN.BBOX_REG:
            blobs['bbox_targets'] = bbox_targets_blob
            blobs['bbox_inside_weights'] = bbox_inside_blob
            blobs['bbox_outside_weights'] = \
                np.array(bbox_inside_blob > 0).astype(np.float32)

    return blobs

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        #fg_inds=[int(x) for x in fg_inds if x]
        fg_inds = npr.choice(
                fg_inds, size=int(fg_rois_per_this_image), replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
                bg_inds, size=int(bg_rois_per_this_image), replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[int(fg_rois_per_this_image):] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets'][keep_inds, :], num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_inside_weights

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        ind = int(ind)
        cls = clss[ind]
        start = int(4 * cls)
        end = (start + 4)
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in range(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print ('class: ', cls, ' overlap: ', overlaps[i])
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
