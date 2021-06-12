import numpy as np
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from utils import (swap_xy, convert_to_xywh, convert_to_corners, scaled_iou,
                   display_sample, random_flip_horizontal, resize_and_pad_image,
                   dataset_split
                   )

#--------------------------------------------------------------------------------
# anchors : for 416x416 image input
##ANCHORS = np.array([[[ 10, 13], [ 16,  30], [ 33,  23]],
##                    [[ 30, 61], [ 62,  45], [ 59, 119]],
##                    [[116, 90], [156, 198], [373, 326]]], np.float32)
### baseline anchors :  for unit cell size
### anchors[i] = ANCHORS[i]/strides[i] where i = 0,1,2 
##anchors = np.array([[[ 1.25 , 1.625 ], [ 2.0  , 3.75  ], [  4.125  ,  2.875 ]],
##                    [[ 1.875, 3.8125], [ 3.875, 2.8125], [  3.6875 ,  7.4375]],
##                    [[ 3.625, 2.8125], [ 4.875, 6.1875], [ 11.65625, 10.1875]]])
##
##strides = np.array([8, 16, 32]) # a cell size for each scale : 416/[52,16,13] = [8, 16, 32]
##for i in range(3):
##    print(anchors[i]*strides[i] == ANCHORS[i]) 

NUM_CLASSES = 20
class LabelEncoder(object):
    """preprocess dataset : """
    def __init__(self, dataset, num_samples, batch_size):
        super(LabelEncoder, self).__init__()
        self.input_size = [416, 416]  
        self.strides = tf.constant([8, 16, 32], dtype=tf.float32)
        self.anchor_per_scale = 3 
        # dataset is expected to have batch slices like 
        # e.g list = ['train[0:100]', 'train[100:200]'. ....]
        self.dataset = dataset
        self.num_samples = num_samples # not always equal to num_all_dps because of UnicodeDecodeError
        self.batch_size  = batch_size
        self.num_batchs = int(np.ceil(self.num_samples/self.batch_size))
        self.num_all_dps = 0        # the number of all data points successfully stored in target
        self.max_box_per_scale = 150# max number of boxes for a given scale
        self.match_iou = 0.3
        self.i = 0  # idx for batch
        # anchor (width, height) noramlized by the strides 
        self.anchors = tf.constant([[[ 1.25 , 1.625 ], [ 2.0  , 3.75  ], [  4.125  ,  2.875 ]],
                                    [[ 1.875, 3.8125], [ 3.875, 2.8125], [  3.6875 ,  7.4375]],
                                    [[ 3.625, 2.8125], [ 4.875, 6.1875], [ 11.65625, 10.1875]]], dtype=tf.float32)
       
    def _preprocess_sample(self, sample):
        # parse data sample and augment on image and boxes
        oimage = sample['image']                    # original image
        boxes = swap_xy(sample['objects']['bbox'])  # original box 
        class_ids = sample["objects"]['label']
        image, boxes = random_flip_horizontal(oimage, boxes)# <dtype: 'uint8'> <dtype: 'float32'>
        #---------------------------------------------------------------
        # resize image 416 with padding and relocate boxes accordingly
        # offsets holds info on how much boxes moved by
        # jitter=true for training
        padded_img, new_size, ratio, offsets = resize_and_pad_image(image,jitter=True)
        #print('offsets :', offsets)
        offsets = tf.cast(offsets, tf.float32)
        boxes = tf.stack([boxes[:,0]*new_size[1]+offsets[1], boxes[:,1]*new_size[0]+offsets[0],
                          boxes[:,2]*new_size[1]+offsets[1], boxes[:,3]*new_size[0]+offsets[0],], axis=-1,)
        #---------------------------------------
        # xywh means [x_center, y_center, w, h]
        # tf.tensor.shape = ( n_objs, 4 )
        gt_boxes = convert_to_xywh(boxes)
        return padded_img/255., gt_boxes, class_ids

    def _get_anchor_boxes(self, i, box_scaled):
        """i( 0,1,2 ) is an idex for a grid scale:"""
        anchor_boxes = np.zeros((self.anchor_per_scale, 4 )) # 3 anchor boxes for i
        anchor_boxes[:, 0:2] = np.floor(box_scaled[i, 0:2]).astype(np.int32) + 0.5
        anchor_boxes[:, 2:4] = self.anchors[i]  # use unit anchors ( scaled by strides)
        return tf.cast(anchor_boxes, tf.float32)


    def _encode_sample(self, gt_boxes, class_ids):
        #--------------------------------------------------------------------------------------
        # strides = [8, 16, 32] : cell sizes of each grid -->  416/[52,26,13] = [8, 16, 32]
        # divide 416x416 image size with strides -> generates 52 grid, 26 grid, 13 grid
        grid_sizes = tf.cast(self.input_size[0] // self.strides, dtype=tf.int32)
        labels = [np.zeros((grid_sizes[i], grid_sizes[i], self.anchor_per_scale, 5+NUM_CLASSES)) for i in range(3)]
        # box container : store all boxes 
        boxes_all = [np.zeros((self.max_box_per_scale, 4)) for _ in range(3)]
        box_count = np.zeros((3,))  # for 3 grid scales
        #--------------------------------------------------------------------------------------
        # box       : [x_center, y_center, w, h]
        # gt_boxes  : tf.tensor.shape = (num_boxes, 4)
        # class_ids : tf.tensor.shape = (num_objs, )
        for box, cls in zip(gt_boxes, class_ids):
            #print('box.dtype :', box.dtype)
            # Dividing a box_xywh by 3 strides translates (x,y,w,h) 
            # into those in terms of the 3 grids : 52x52, 26x26, 13x13 
            box_scaled = box[ tf.newaxis, :] / self.strides[:, tf.newaxis]
            one_hot = tf.one_hot(cls, NUM_CLASSES)
            uniform_dist = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
            delta = 0.01
            smooth_onehot = one_hot*(1 - delta) + delta * uniform_dist
            IOUS = [] ; detect = False  # whether or not anchor_mask detects objs
            for i in range(3): # 3 grid scales
                anchor_boxes = self._get_anchor_boxes(i, box_scaled)
                # iou vals for 3 anchor boxes
                iou = scaled_iou(box_scaled[i][tf.newaxis, :], anchor_boxes)
                IOUS.append(iou)
                anchor_mask = tf.squeeze(iou > self.match_iou) # since we want (3,)  squeeze iou.shape = (1,3) 
                if np.any(anchor_mask):
                    x, y = np.floor(box_scaled[i, 0:2]).astype(np.int32)# (x,y) position of box for i-th scale
                    x = np.clip(x, 0, grid_sizes[i]-1)     
                    y = np.clip(y, 0, grid_sizes[i]-1)     
                    # x and y are switched because of the meshgrid created in decode fnc,
                    # we're gonna compute loss with pred_box from decode and true_box from here
                    labels[i][y, x, anchor_mask, :] = 0          #  initialize label bucket
                    labels[i][y, x, anchor_mask, 0:4]  = box     #  original box (not box_scaled)
                    labels[i][y, x, anchor_mask, 4:5]  = 1.0     #  objectness
                    labels[i][y, x, anchor_mask, 5:]   = smooth_onehot #  NUM_CLASSES class_ids

                    box_idx = int(box_count[i] % self.max_box_per_scale)
                    boxes_all[i][box_idx, :4] = box
                    box_count[i] += 1   # i for grid scale
                    detect = True       # if an anchor of any scale matches gt_box with iou > match_iou

            # If a box is not captured by any grid scale of anchor boxes with iou > self.match_iou
            # Find the max iou and the corresponding (x, y), and store the box there
            if not detect:
                # pick one out of 9 idxs : [ 9 iou vals ]
                best_anchor_idx = np.argmax(np.array(IOUS).reshape(-1), axis=-1)
                grid_idx = int(best_anchor_idx / self.anchor_per_scale) #0,1,2 which grid? 52, 26, 13
                anchor_idx = int(best_anchor_idx % self.anchor_per_scale)
                x, y = np.floor(box_scaled[grid_idx, 0:2]).astype(np.int32)
                x = np.clip(x, 0, grid_sizes[i]-1)     
                y = np.clip(y, 0, grid_sizes[i]-1)     
                labels[grid_idx][y, x, anchor_idx, :] = 0      #  initialize labels bucket
                labels[grid_idx][y, x, anchor_idx, 0:4] = box
                labels[grid_idx][y, x, anchor_idx, 4:5] = 1.0
                labels[grid_idx][y, x, anchor_idx, 5:]  = smooth_onehot

                box_idx = int(box_count[grid_idx] % self.max_box_per_scale)
                boxes_all[grid_idx][box_idx, :4] = box
                box_count[grid_idx] += 1   

        return labels, tf.cast(boxes_all, tf.float32)

    def __iter__(self):
        return self

    def __next__(self):
        fails = 0
        # batch_size : number of samples per batch
        # without specification of np.float32, dtype will be np.float64 by default
        batch_images = np.zeros((self.batch_size, 416, 416, 3), dtype=np.float32)
        batch_label_box52 = np.zeros((self.batch_size, 52, 52, 3, 5+NUM_CLASSES))
        batch_label_box26 = np.zeros((self.batch_size, 26, 26, 3, 5+NUM_CLASSES))
        batch_label_box13 = np.zeros((self.batch_size, 13, 13, 3, 5+NUM_CLASSES))

        batch_box52 = np.zeros((self.batch_size, self.max_box_per_scale, 4),dtype=np.float32) 
        batch_box26 = np.zeros((self.batch_size, self.max_box_per_scale, 4),dtype=np.float32) 
        batch_box13 = np.zeros((self.batch_size, self.max_box_per_scale, 4),dtype=np.float32) 

        if self.i < self.num_batchs:
            batch = self.dataset[self.i].as_numpy_iterator()
            num_iters = 0 ; num_dps = 0
            # iterate over each batch
            while num_iters < len(self.dataset[self.i]):
                try :
                    num_iters += 1
                    sample = batch.next()
                    image, boxes, class_ids = self._preprocess_sample(sample)
                    labels, boxes_all = self._encode_sample(boxes, class_ids)
                    label52, label26, label13 = labels
                    batch_images[num_dps, :, :, :] = image
                    batch_label_box52[num_dps, :, :, :, :] = label52
                    batch_label_box26[num_dps, :, :, :, :] = label26
                    batch_label_box13[num_dps, :, :, :, :] = label13
                    box52, box26, box13 = boxes_all
                    batch_box52[num_dps, :, :] = box52
                    batch_box26[num_dps, :, :] = box26
                    batch_box13[num_dps, :, :] = box13
                    num_dps += 1
                    #print('sample[image].shape :', sample['image'].shape)
                except UnicodeDecodeError:
                    fails += 1
                    continue
            #print('self.num_all_dps :', self.num_all_dps)
            self.num_all_dps += num_dps

            ytrue52 = batch_label_box52, batch_box52
            ytrue26 = batch_label_box26, batch_box26
            ytrue13 = batch_label_box13, batch_box13
            self.i += 1
            return batch_images, (ytrue52, ytrue26, ytrue13)
        else:
            self.i = 0
            raise StopIteration

    def __len__(self):
        #since the class is supposed to be used in a loops
        return self.num_batchs



#----------------------------------------------------------------------------
#
#   Create data_split (batch) for training
#
#----------------------------------------------------------------------------
# start : start of data point
# num_samples : how many data points
# batch_size  : the number of batchs splits num_samples  
##opt ='train'; start = 2000 ; num_samples = 20 ; batch_size = 8
##train_split = dataset_split(start, num_samples, batch_size, opt)
##print('train_split :', train_split)
##train_ds = tfds.load("voc", split=train_split, data_dir="data")
##print('num_batchs :', len(train_ds))
##
###----------------------------------------------------------------------------
###   Test _preprocess_sample method
###
##examples = LabelEncoder(train_ds, num_samples, batch_size)
##batch0 = train_ds[1]
##batch0 = batch0.as_numpy_iterator()
## #sample = batch0.next()
##for sample in batch0:
##    image, gt_boxes, class_ids = examples._preprocess_sample(sample) # just test
##    display_sample(image, gt_boxes, class_ids)


##EPOCHS = 3
##for epoch in range(1, EPOCHS+1):
##    print('-------------------------')
##    print('epoch :', epoch)
##    for j , batch in enumerate(examples):
##        print('j :', j, ' len(batch) :', len(batch))
##        x_batch, y_batch = batch
##        print(x_batch.shape)

##for name in dir():
##    if not name.startswith('_'):
##        del globals()[name]

