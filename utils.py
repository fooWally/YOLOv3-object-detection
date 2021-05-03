import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def dataset_split(start, num_samples, batch_size, opt):
    stop = start + num_samples
    num_batchs = np.ceil(num_samples/batch_size)
    num_batchs = int(num_batchs)
    split = []
    for i in range(start, stop, batch_size):
        if (i+batch_size) > stop:
            string = f'{opt}[{i}:{stop}]'
        else:
            string = f'{opt}[{i}:{i+batch_size}]'
        split.append(string)
    #print('split :', split)
    return split    
#------------------------------------------------------------------------
#   TEST dataset_split
#
##opt ='train'; start = 0 ; num_samples = 10 ; batch_size = 2
##train_split = dataset_split(start, num_samples, batch_size, opt)
##opt ='validation'
##val_split = dataset_split(start, num_samples, batch_size, opt)
##split = [train_split, val_split]
##print('split :', split)
##train_ds, val_ds = tfds.load("coco/2017", split=split, data_dir="data")
##print(len(train_ds))
##print(len(val_ds))

#-------------------------------------------------------------------------
#https://keras.io/examples/vision/retinanet/
def swap_xy(boxes):
    return tf.stack([boxes[:, 1], boxes[:, 0],
                     boxes[:, 3], boxes[:, 2]], axis=-1)

def convert_to_xywh(boxes):
    return tf.concat([(boxes[..., :2] + boxes[..., 2:])/ 2.0,
                      boxes[..., 2:] - boxes[..., :2]], axis=-1)

def convert_to_corners(boxes):
    return tf.concat([boxes[..., :2] - boxes[..., 2:]/2.0,
                      boxes[..., :2] + boxes[..., 2:]/2.0], axis=-1)

def random_flip_horizontal(image, boxes):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack([1 - boxes[:, 2], boxes[:, 1],
                          1 - boxes[:, 0], boxes[:, 3]], axis=-1)
    return image, boxes

def resize_and_pad_image(oimage, jitter=True):
    # Jitter is True for training and False for prediction
    oimage_size = tf.cast(tf.shape(oimage)[:2], tf.float32)
    target_size= [416, 416] # size of the final image
    #--------------------------------------------------
    # Jitter oimage : pick a width for oimage
    # e.g oimage_shape = [462. 640.]
    if jitter:
        jitter = tf.random.uniform((), 360, 416, dtype=tf.float32)
    else:
        jitter = 416.0
    max_side = tf.reduce_max(oimage_size) # reduce 640 to 416 : find max_side and reduce it to 416
    ratio = jitter/max_side
    # e.g oimage_shape*ratio = [462. 640.]* jitter/640  
    #img_shape = oimage_shape * ratio
    new_size = tf.floor(oimage_size * ratio)
    #-------------------------------------------------------
    # resize oimage by img_shape with the ratio of oimage w/h kept
    image = tf.image.resize(oimage, tf.cast( new_size, dtype=tf.int32)) # might be the cause of deviated bbox from an object
    offset_h, offset_w = tf.cast( tf.floor((target_size - new_size)/2.0), tf.int32)
    padded_img = tf.image.pad_to_bounding_box(image, offset_h, offset_w, target_size[0], target_size[1])  
    #-------------------------------------------------------#
    # padded_img/255.0 SHOULD NOT BE INCLUDED IN TRAINING   #
    #padded_img = padded_img/255.0 ,                        #
    #which will display padded_img entirely black           #
    #-------------------------------------------------------#
    offsets = offset_h, offset_w
    return padded_img , new_size, ratio, offsets



def scaled_iou(boxes1, boxes2):
    """" compute iou of scaled_boxes_xywh and anchor_xywh """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    #----------------------------------------------------------------
    # convert_to_corners: xywh --> xmin,ymin,xmax,ymax
    boxes1 = convert_to_corners(boxes1)
    boxes2 = convert_to_corners(boxes2)
    #----------------------------------------------------------------
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    return inter_area / union_area

def compute_iou(boxes1, boxes2):
    # use this for computing iou of boxes in high dimensions: e.g compute_loss
    """ takes boxes of format: [x, y, width, height]
        supposed to work for (N, 4) and (M, 4) where N, M  are the number of boxes
        returns pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
        works for boxes of high dimensions (batch_size, 52, 52, 3, 1, 4)
        and (batch_size, 1, 1, 1, 150, 4) """
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = np.maximum(1.0*inter_area / union_area, np.finfo(np.float32).eps)
    return iou



def load_weights(model, weights_file):
    """
    #=================================================
    #   Copyright (C) 2019 * Ltd. All rights reserved.
    #   Editor      : VIM
    #   File name   : utils.py
    #   Author      : YunYang1994
    #   Created date: 2019-07-12 01:33:38
    #=================================================
    """
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(75):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in [58, 66, 74]:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in [58, 66, 74]:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()



def display_sample(image, boxes, class_ids):
    """For an object, a box is of format [xc,yc,w,h]
        -> boxes.shape (num_boxes, 4) """
    scores = tf.ones(len(class_ids))
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca() ; color = 'y'
    for box, _cls, score in zip(boxes, class_ids, scores):
        text = "{}: {:.2f}".format(_cls, score)
        xc, yc, w, h = box  # x_center, y_center, w, h
        x = xc - w/2.0 ; y = yc - h/2.0
        patch = plt.Rectangle([x, y], w, h, fill=False, edgecolor=color, linewidth=0.5)
        ax.add_patch(patch)
        ax.text( x, y, text, bbox={"facecolor": color, "alpha": 0.4},
                 clip_box=ax.clipbox, clip_on=True, fontsize=8.0 )
    plt.show()

