import numpy as np
import tensorflow as tf
import utils
import colorsys
from random import randint
import matplotlib.pyplot as plt

#----------------------------------------------------------------
#   Inference model and the weights as inputs
def detections(inference_model, image_data, oimage_size, score_threshold = 0.5,
               iou_threshold = 0.6, max_detections = 20):
    image_data = tf.cast(image_data[tf.newaxis, ...], tf.float32)
    pred_bbox = inference_model.predict(image_data)  # image_data : 416x416
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    print('pred_bbox.shape :', pred_bbox.shape)
    #pred_bbox = np.array(pred_bbox) # (10647, 5+num_classes)
    pred_xywh = pred_bbox[:, 0:4]   # (10647,  4)
    pred_conf = pred_bbox[:, 4]     # (10647,  1)
    pred_prob = pred_bbox[:, 5:]    # (10647, num_classes)

    # 1.select idx of max entry in each row :
    #   that is cls_id because we are using one-hot encoding
    cls_idxs = tf.argmax(pred_prob, axis=1) # not 0-axis
    #print('cls_idxs.shape :', cls_idxs.shape)# (10647,)

    # 2.pick cls_scores according to the idx found from 1
    #   we need cls_scores for tf.image.non_max_suppression
    idxs = tf.cast(cls_idxs, tf.int32)
    idx_full = tf.range(len(idxs))[...,tf.newaxis]
    idx_pairs = [idx_full, idxs[...,tf.newaxis]]
    idx_stack = tf.stack(idx_pairs, axis=2)
    # cls_scores is the max score whose idx is found by tf.argmax
    pred_prob = tf.squeeze(tf.gather_nd(pred_prob, idx_stack)) # (10647,)
    # Define score
    scores = pred_conf * pred_prob # pred_conf * pred_prob 
    # need [xmin,ymin,xmax,ymax] format for tf.image.non_max_suppression
    boxes = utils.convert_to_corners(pred_xywh)
    score_mask = scores > score_threshold
    boxes, scores, classes = boxes[score_mask], scores[score_mask], cls_idxs[score_mask]

    nms_boxes, nms_scores, nms_classes = nms_process(boxes, scores, classes, max_detections, iou_threshold) 
    print(nms_boxes)
    print(nms_classes)
    # [xmin, ymin, xmax, ymax] -> [xmin_org, ymin_org, xmax_org, ymax_org]
    nms_oboxes = scale_nms_boxes(nms_boxes, oimage_size) # scale boxes back to original
    bboxes = np.concatenate([nms_oboxes, nms_scores[:, np.newaxis], nms_classes[:, np.newaxis]], axis=-1)
    return bboxes

#----------------------------------------------------------------
# 3. Using non_max_suppression,
#   get nms_idxs to find pred_boxes that has iou > iou_threshold
#   pred_boxes have all the info about objects that are found
def nms_process(boxes, scores, classes, max_detections = 20, iou_threshold = 0.6):
    ### Boxes here are in the corner format: [xmin,ymin,xmax,ymax]
    nms_idxs = tf.image.non_max_suppression(boxes, scores, max_detections, iou_threshold)
    #print('nms_idxs :',nms_idxs) # tf.int32
    nms_boxes = tf.gather(boxes, nms_idxs) # tf.float32
    nms_scores = tf.gather(scores, nms_idxs)
    nms_classes = tf.gather(classes, nms_idxs)
    return nms_boxes, nms_scores, nms_classes

def scale_nms_boxes(nms_boxes, oimage_size):
    oh, ow = oimage_size
    resize_ratio = min(416 / ow, 416 / oh)
    dw = (416 - resize_ratio * ow) / 2
    dh = (416 - resize_ratio * oh) / 2
    # 0::2 = xmin, xmax --> width, 1::2 = ymin, ymax --> height
    nms_boxes = np.array(nms_boxes)
    nms_boxes[:, 0::2] = 1.0 * (nms_boxes[:, 0::2] - dw) / resize_ratio
    nms_boxes[:, 1::2] = 1.0 * (nms_boxes[:, 1::2] - dh) / resize_ratio
    nms_oboxes = np.concatenate([np.maximum(nms_boxes[:, :2], [0, 0]),
                                 np.minimum(nms_boxes[:, 2:], [ow - 1, oh - 1])], axis=-1)
    return nms_oboxes


def display_detections(oimage, bboxes):
    """
    bboxes: (N, [xmin, ymin, xmax, ymax, score, cls_id])
    """
    image = np.array(oimage, dtype=np.uint8)
    # ms coco num_classes = 80
    #classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    # voc2007 num_classes = 20
    classes = ["aeroplane","bicycle","bird","boat","bottle", "bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor",]

    num_classes = len(classes) 
    colors = []
    for i in range(num_classes):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
        
    plt.figure(figsize=(6, 4))
    plt.imshow(image)
    ax = plt.gca()
    for bbox in bboxes:
        xmin,ymin,xmax,ymax = bbox[:4]
        score = bbox[4]
        class_id = int(bbox[5])
        bbox_color = colors[class_id]
        text = "{}:{:.2f}".format(classes[class_id], score)
        w = abs(xmin-xmax); h = abs(ymin-ymax)  # Rectangle takes xmin,ymin, w, h
        patch = plt.Rectangle([xmin, ymin], w, h, fill=False, edgecolor=bbox_color, linewidth=0.5)
        ax.add_patch(patch)
        box_dict={"facecolor": bbox_color, "alpha": 0.4,}
        ax.text(xmin, ymin, text, bbox=box_dict, fontsize=7.0)

    plt.show()
    

