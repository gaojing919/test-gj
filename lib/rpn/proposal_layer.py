# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
from .generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms

def computScale(bottom):
    anchor_scale = np.zeros(2)
    pre_scale = bottom.reshape(40)
    #pre_scale = bottom[4].data.reshape(60)
    #print(pre_scale)
    #plt.plot(range(60),pre_scale)
    #plt.show()
    #moxing average smoothed
    #m = np.mean(pre_scale)
    for i in range(4):
        m=np.mean(pre_scale[10*i:10*i+10])
        pre_scale[10*i:10*i+10] = pre_scale[10*i:10*i+10]-m
    #pre_scale[:] = pre_scale[:] - m
    #print(pre_scale)
    #plt.plot(range(60),pre_scale)
    #plt.show()
    #1D NMS
    mark = np.zeros(40)
    while(0 in mark):
        index = np.array(np.where(mark==0)[0])
        #print(index)
        #index.reshape(index[0].size)
        #print(index)
        unhandle = pre_scale[index]
        maxpostion = np.argmax(unhandle)
        maxscale_index = index[maxpostion]
        mark[maxscale_index] = 1
        for i in range(maxscale_index-3,maxscale_index+4):
            if(i>=0 and i<40):
                if(mark[i]==0):
                    mark[i]=-1
        #print(mark)
        #print(maxpostion)
        #print(index[maxpostion])
        #print(pre_scale[maxpostion])
        #print(index)
        #print(pre_scale)
        #print(unhandle)
        #break
    max_index = np.array(np.where(mark==1)[0])
    max_scale = pre_scale[max_index]
    #print(len(max_scale))
    #print("gj_")
    #print(max_index)
    #print(max_scale)
    while(0 in anchor_scale):
        anchor_index = np.array(np.where(anchor_scale==0)[0])
        s = np.argmax(max_scale)
        #print(s)
        #print(max_index[s])
        if(max_scale[s]==-1):
            break
        max_scale[s] = -1
        #a_s = 4*2**int(max_index[s]/10)
        a_s = int(max_index[s])+1
        #if(abs(max_index[s]-a_s)>abs(max_index[s]-a_s/2)):
        #    a_s = a_s/2
        '''
        if(a_s<4):
            a_s = 4
        if(a_s>32):
            a_s = 32
        '''
        if(a_s<1):
            a_s =1
        if(a_s>40):
            a_s = 40
        
        #print(a_s)
        if(a_s not in anchor_scale):
            anchor_scale[anchor_index[0]] = a_s
    while(0 in anchor_scale):
        anchor_index = np.array(np.where(anchor_scale==0)[0])
        if(8 not in anchor_scale):
            anchor_scale[anchor_index[0]] = 8
        if(16 not in anchor_scale):
            anchor_scale[anchor_index[0]] = 16
        #if(64 not in anchor_scale):
        #    anchor_scale[anchor_index[0]] = 64
    #minscale = np.argmin(anchor_scale)
    #anchor bu neng quan da yu 32
    #if(anchor_scale[minscale]>32):
    #    anchor_scale[minscale] = 32
    anchor_scale.sort()
    print(anchor_scale)
    return anchor_scale

DEBUG = False

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._feat_stride = layer_params['feat_stride']
        anchor_scales = layer_params.get('scales', (8,16))
        #print("gjanchor_scales")
        #print(anchor_scales)
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]

        if DEBUG:
            print ('feat_stride: {}'.format(self._feat_stride))
            print ('anchors:')
            print (self._anchors)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)
        anchor_scale = computScale(bottom[3].data)
        #anchor_scale = np.array([8,32])


        new_anchors = generate_anchors(scales=anchor_scale)

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, self._num_anchors:, :, :]
        bbox_deltas = bottom[1].data
        im_info = bottom[2].data[0, :]

        if DEBUG:
            print ('im_size: ({}, {})'.format(im_info[0], im_info[1]))
            print ('scale: {}'.format(im_info[2]))

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        if DEBUG:
            print ('score map size: {}'.format(scores.shape))

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        #print("A")
        #print(A)
        K = shifts.shape[0]
        anchors = new_anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        #print("proposals")
        #print(len(proposals))
        keep = _filter_boxes(proposals, min_size * im_info[2])
        #print("keep")
        #print(keep)
        proposals = proposals[keep, :]
        scores = scores[keep]
        print("gjscores")
        print(scores)

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

        # [Optional] output scores blob
        if len(top) > 1:
            top[1].reshape(*(scores.shape))
            top[1].data[...] = scores
        print("okanchor")

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
