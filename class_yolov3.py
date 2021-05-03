import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras import layers, Model
import utils
clear_session()

# Batchnormalization already includes the addition of the bias term.
# Recap that BatchNorm is already: gamma * normalized(x) + bias
# So there is no need (and it makes no sense) to add another bias term
# in the convolution layer.

class Darknet53(layers.Layer):
    def __init__(self):
        super(Darknet53, self).__init__(name='Darknet53')
    #------------------------------------------------------------------------
    #   common layer components for Darknet53 and for YOLOv3
    def _conv2d(self, x, filters, kernel_size, downsample=False, activate=True, bn=True):
        if downsample:
            # To reduce size of x : 256 --> 128, introduce a zero-padding
            x = layers.ZeroPadding2D(((1,0),(1,0)))(x)
            padding = 'valid'; strides = 2
        else:
            padding = 'same'; strides = 1
            
        conv = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,use_bias=not bn,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                             bias_initializer=tf.constant_initializer(0.))(x)
        if bn: conv = layers.BatchNormalization()(conv)
        if activate == True : conv = layers.LeakyReLU(alpha=0.1)(conv)
        return conv

    def _residual_block(self, x, filters1, filters2):
        stored_x = x
        conv = self._conv2d(   x, filters1, kernel_size=1)
        conv = self._conv2d(conv, filters2, kernel_size=3)
        residual_x = stored_x + conv
        return residual_x
    #------------------------------------------------------------------------

    def __call__(self, inputs):
        x = self._conv2d(inputs, 32, 3)             # output size: 256x256
        x = self._conv2d(x, 64, 3, downsample=True) # output size: 128x128
        #--------------------------------------
        #   32: filters for 1st conv layer
        #   64: filters for 2nd conv layer
        x = self._residual_block(x, 32, 64) # output size: 128x128 

        x = self._conv2d(x, 128, 3, downsample=True) #output size: 64x64 downsampled from 128x128
        #--------------------------------------
        #   ( 64: filters for 1st conv layer
        #    128: filters for 2nd conv layer) x 2
        for i in range(2):
            x = self._residual_block(x, 64, 128)
            # final output size: 64x64
            
        x = self._conv2d(x, 256, 3, downsample=True) #output size: 32x32 downsampled 
        #---------------------------------------
        #   (128: filters for 1st conv layer
        #    256: filters for 2nd conv layer) x 8
        for i in range(8):
            x = self._residual_block(x, 128, 256)   #output size: 32x32
        residual_1 = x # 52X52

        x = self._conv2d(x, 512, 3, downsample=True) #output size: 16x16 downsampled
        #---------------------------------------
        #   (256: filters for 1st conv layer
        #    512: filters for 2nd conv layer) x 8
        for i in range(8):
            x = self._residual_block(x, 256, 512)  #output size: 16x16
        residual_2 = x # 26X26

        x = self._conv2d(x, 1024, 3, downsample=True) #output size: 8x8 downsampled
        #---------------------------------------
        #   (512 : filters for 1st conv layer
        #    1024: filters for 2nd conv layer) x 4
        for i in range(4):
            x = self._residual_block(x, 512, 1024)  #output size: 8x8
        return [residual_1, residual_2, x]

#######################
#
#   TEST darknet53
#
##input_size = 416 
##input_shape = [input_size, input_size, 3]
##inputs = layers.Input(shape=input_shape)
##print('inputs =', inputs)
##
##darknet = Darknet53()
##outputs = darknet(inputs)
##print('outputs =', outputs)
##model = Model(inputs, outputs)
##model.summary()


# tf.keras.layers.Layer : the class from which all layers inherit.
NUM_CLASSES = 20    # voc/2007 dataset
##NUM_CLASSES = 80  # mscoco dataset 
class YOLOv3(layers.Layer):
    def __init__(self):
        super(YOLOv3, self).__init__(name='YOLOv3')
        # 3 anchor_masks x(4+1+NUM_CLASSES) : NUM_CLASSES =80 for ms-coco dataset
        self.num_filters = 3 * (4+1+NUM_CLASSES)
        self.darknet = Darknet53()
    #------------------------------------------------------------------------
    #   common layer components for Darknet53 and for YOLOv3
    def _conv2d(self, x, filters, kernel_size, downsample=False, activate=True, bn=True):
        if downsample:
            # To reduce size of x : 256 --> 128, introduce a zero-padding
            x = layers.ZeroPadding2D(((1,0),(1,0)))(x)
            padding = 'valid'; strides = 2
        else:
            padding = 'same'; strides = 1
            
        conv = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,use_bias=not bn,
                             kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                             kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                             bias_initializer=tf.constant_initializer(0.))(x)
        if bn: conv = layers.BatchNormalization()(conv)
        if activate == True : conv = layers.LeakyReLU(alpha=0.1)(conv)
        return conv

    def _residual_block(self, x, filters1, filters2):
        stored_x = x
        conv = self._conv2d(   x, filters1, kernel_size=1)
        conv = self._conv2d(conv, filters2, kernel_size=3)
        residual_x = stored_x + conv
        return residual_x
    #------------------------------------------------------------------------

    def __call__(self, inputs):
        # darknet = Darknet53()
        x52, x26, x13 = self.darknet(inputs) # 3 features tensors

        for f, k in [(512, 1),(1024, 3),(512, 1),(1024, 3),(512, 1)]:
            x = self._conv2d(x13, f, k)
            x13 = x

        y13 = self._conv2d(x, 1024, 3)
        # activate=False, bn=False because we want raw conv vals from this layer
        y13 = self._conv2d(y13, self.num_filters, 1, activate=False, bn=False)
        
        x = self._conv2d(x, 256, 1)
        x = layers.UpSampling2D(size=(2,2))(x)
        x = layers.Concatenate()([x, x26])
        #print('x :', x)
        for f, k in [(256, 1),(512, 3),(256, 1),(512, 3),(256, 1)]:
            x = self._conv2d(x, f, k)

        y26 = self._conv2d(x, 512, 3)
        # activate=False, bn=False because we want raw conv vals from this layer
        y26 = self._conv2d(y26, self.num_filters, 1, activate=False, bn=False)
        
        x = self._conv2d(x, 128, 1)
        x = layers.UpSampling2D(size=(2,2))(x)
        x = layers.Concatenate()([x, x52])

        for f, k in [(128, 1),(256, 3),(128, 1),(256, 3),(128, 1)]:
            x = self._conv2d(x, f, k)

        y52 = self._conv2d(x, 256, 3)
        # activate=False, bn=False because we want raw conv vals from this layer
        y52 = self._conv2d(y52, self.num_filters, 1, activate=False, bn=False)
        
        return [y52, y26, y13]

######################
#
#   TEST Yolov3()
#
##input_size = 416 
##input_shape = [input_size, input_size, 3]
##inputs = layers.Input(shape=input_shape)
##print('inputs =', inputs)
##
##yolov3 = YOLOv3()
##outputs = yolov3(inputs)
##model = Model(inputs, outputs)
##model.summary()


anchors = tf.constant([[[ 1.25 , 1.625 ], [ 2.0  , 3.75  ], [  4.125  ,  2.875 ]],
                        [[ 1.875, 3.8125], [ 3.875, 2.8125], [  3.6875 ,  7.4375]],
                        [[ 3.625, 2.8125], [ 4.875, 6.1875], [ 11.65625, 10.1875]]], tf.float32)

strides = tf.constant([8, 16, 32], tf.float32) # a cell size for each scale : 416/[52,16,13] = [8, 16, 32]

def decode(conv_output, i=0):
    # conv_output : (N, grid, grid, 3, 5 + NUM_CLASSES) where N : batch_size and 3 = num_anchors_per_grid
    # transform convolution outputs from YOLOv3 detector into YOLO Prediction 
    conv_shape = tf.shape(conv_output) 
    batch_size = conv_shape[0]
    grid_size  = conv_shape[1]
    conv_output = tf.reshape(conv_output, (batch_size, grid_size, grid_size, 3, 5+NUM_CLASSES))

    t_xy = conv_raw_dxdy = conv_output[..., 0:2]
    t_wh = conv_raw_dwdh = conv_output[..., 2:4]
    conv_raw_conf = conv_output[..., 4:5]
    conv_raw_prob = conv_output[..., 5: ]
    #-----------------------------------------------------------------------------
    # Box Prediction: first create the grid of 52x52, 26x26, 13x13 
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    # grid is a list; len(grid) = 2 where grid[0] = x and grid[1] = y
    grid_stack = tf.stack(grid, axis=-1)        # grid_stack.shape = [grid, grid, 2]
    grid = tf.expand_dims(grid_stack, axis=2)   # [gx, gy, 1, 2]
    grid = tf.cast(grid, tf.float32)
    # box prediction equation
    # bx = sigmoid(tx) + cx where (cx,cy) corresponds to grid
    # by = sigmoid(ty) + cy       and sigmoid(tx,ty) are offset vals btw 0 and 1
    # bw = Pw * exp(tw)           multiplied by strides recovering 416 size  
    # bh = Ph * exp(th)
    pred_xy = (tf.sigmoid(t_xy) + grid) * strides[i]
    pred_wh = (tf.exp(t_wh) * anchors[i]) * strides[i]
    pred_box = tf.concat([pred_xy, pred_wh], axis=-1)
    # pred_box : (N, grid, grid, anchor, 4) where 4 = [xc,yc,w,h]
    #-----------------------------------------------------------------------------
    pred_obj = tf.sigmoid(conv_raw_conf) # objectness prediction
    pred_cls = tf.sigmoid(conv_raw_prob) # class prob prediction

    y_pred = tf.concat([pred_box, pred_obj, pred_cls], axis=-1)
    # y_pred : : (N, grid, grid, anchor, 4+1+NUM_CLASSES)
    # where 4 = [xc,yc,w,h], 1= obj, NUM_CLASSES = num_clses in one-hots
    return y_pred


#----------------------------------------------------------------------------
#   TEST : Build a model(inputs, outputs)
#
##input_size = 416 
##input_shape = [input_size, input_size, 3]
##inputs = layers.Input(shape=input_shape)
##print('inputs =', inputs)
##
##yolov3 = YOLOv3()               # yolov3 class instance : calls darknet53 as backbone by default
##yolo_outputs = yolov3(inputs)   # [conv52, conv26, conv13]
##y_preds = []
##for i, conv in enumerate(yolo_outputs):
##    y_pred = decode(conv, i)
##    y_preds.append(y_pred)
##outputs = y_preds
##print('outputs :', outputs)
##model = Model(inputs, outputs)
##model.summary()


#---------------------------------------------
#   y_pred : an output from decode per scale
def compute_loss(y_pred, y_true):
    # target label : y_true per a scale 52, 26, 13, respectively
    label, boxes_all = y_true
    label = tf.cast(label, tf.float32)          # need float32 dtype
    boxes_all = tf.cast(boxes_all, tf.float32)  
    # y_pred = model(X,training) : pred_box xywh
    pred_box, pred_obj, pred_cls = tf.split(y_pred, (4,1,NUM_CLASSES), axis =-1)

    # Take the inverse of sigmoid fnc to obtain conv_raw vals
    conv_raw_obj = - tf.math.log(1.0/(pred_obj + 1e-8) -1)
    conv_raw_cls = - tf.math.log(1.0/(pred_cls + 1e-8) -1)

    true_box, true_obj, true_cls = tf.split(label, (4,1,NUM_CLASSES), axis =-1)
    #--------------------------------------------------------------------------
    # classification loss : uses labels = y_true, logits = raw_vals
    # objectness indicator: 1_obj = true_obj
    class_loss = true_obj * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_cls, logits=conv_raw_cls)
    class_loss = tf.reduce_mean(tf.reduce_sum(class_loss, axis=[1,2,3,4]))

    #--------------------------------------------------------------------------
    # Minimize localization loss to determine where a prediction box
    # of an object should be located
    # box_loss_scale : increases the loss for a small box, so that large box
    # effect doesn't dominate small loss.
    # IOU loss or BOX_xywh ?
    iou = utils.compute_iou(pred_box, true_box)
    iou = tf.expand_dims(iou, axis=-1)
    box_loss_scale = 2.0 - 1.0*true_box[...,2:3] * true_box[...,3:4]/(416.0**2)
    iou_loss = true_obj * tf.cast(box_loss_scale*(1.0 - iou), tf.float32)
    iou_loss = tf.reduce_mean(tf.reduce_sum(iou_loss, axis=[1,2,3,4]))

    #--------------------------------------------------------------------------
    # IOU of pred_box against boxes_all 
    # confidence loss for objectness and non-objectness(background):
    # 1. ignore_threshold = 0.5
    # 2. non-objectness indicator : 1_noobj = (1.0 - true_obj)
    #    bgd_conf = 1_noobj*(max_ious < ignore_threshold)
    # 3. focal_loss : (true_obj - pred_obj)**2 *sigmoid_cross_entropy
    ignore_threshold = 0.5
    #print('iou thresh ok')
    #print('boxes_all.dtype :', boxes_all.dtype)
    #print('tf.shape(pred_box) :', tf.shape(pred_box[:, :, :, :, tf.newaxis, :]))
    #print('tf.shape(boxes_all) :', tf.shape(boxes_all[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :]))
    IOUS = utils.compute_iou(pred_box[:, :, :, :, tf.newaxis, :], boxes_all[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(IOUS, axis=-1), axis=-1)
    ignore_mask =  tf.cast( max_iou < ignore_threshold, tf.float32 )
    bgd_conf = (1.0 - true_obj) * ignore_mask
    focal_loss = tf.pow(true_obj- pred_obj, 2)* tf.nn.sigmoid_cross_entropy_with_logits(labels=true_obj, logits=conv_raw_obj)

    conf_loss = ( true_obj + bgd_conf )* focal_loss
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    #print('class_loss :', class_loss)
    #print('iou_loss :', iou_loss)
    #print('conf_loss :', conf_loss)
    return [class_loss , iou_loss ,conf_loss]


#---------------------------------------------
#   y_pred : an output from decode per scale
##def compute_loss(y_pred, y_true):
##    #
##    # target label : y_true per a scale 52, 26, 13, respectively
##    label, boxes_all = y_true
##    label = tf.cast(label, tf.float32)          # need float32 dtype
##    boxes_all = tf.cast(boxes_all, tf.float32)  
##    # y_pred = model(X,training) : pred_box xywh
##    # (N, grid, grid, 3, 4+1+num_classes)
##    pred_box, pred_obj, pred_cls = tf.split(y_pred, (4,1,NUM_CLASSES), axis =-1)
##    # Take the inverse of sigmoid fnc to obtain conv_raw vals for cross_entropy_with_logits
##    conv_raw_obj = - tf.math.log(1.0/(pred_obj + 1e-8) -1)
##    conv_raw_cls = - tf.math.log(1.0/(pred_cls + 1e-8) -1)
##
##    true_box, true_obj, true_cls = tf.split(label, (4,1,NUM_CLASSES), axis =-1)
##
##    #--------------------------------------------------------------------------
##    # classification loss : uses labels = y_true, logits = raw_vals
##    # objectness indicator: 1_obj = true_obj
##    # (N, grid, grid, 3, num_classes)
##    cross_entropy_cls = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_cls, logits=conv_raw_cls)
##    class_loss = true_obj * cross_entropy_cls
##    class_loss = tf.reduce_mean(tf.reduce_sum(class_loss, axis=[1,2,3,4]))
##
##    #--------------------------------------------------------------------------
##    # Minimize localization loss to determine where a prediction box
##    # of an object should be located
##    # box_loss_scale : increases the loss for a small box, so that large box
##    # effect doesn't dominate small loss.
##    pred_xy = pred_box[..., 0:2] #(N, grid, grid, 3, 2)
##    pred_wh = pred_box[..., 2:4]
##    true_xy = true_box[..., 0:2] #(N, grid, grid, 3, 2)
##    true_wh = true_box[..., 2:4]
##    
##    box_loss_scale = 2.0 - 1.0 * true_wh[...,0] * true_wh[...,1]/(416.0**2)# #(N, grid, grid, 3) ; area = w * h
##    obj_mask = tf.squeeze(true_obj, axis=-1)   # (N, grid, grid, 3)
##    xy_loss1 = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
##    wh_loss1 = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(tf.sqrt(true_wh) - tf.sqrt(pred_wh)), axis=-1)#/(416.0**2) too big
##    #print('xy_loss1.shape :', xy_loss1.shape) # (N, grid, grid, 3)
##    xy_loss = tf.reduce_mean(tf.reduce_sum(xy_loss1, axis=(1, 2, 3))) #tf.reduce_mean over batch_size
##    wh_loss = tf.reduce_mean(tf.reduce_sum(wh_loss1, axis=(1, 2, 3)))
##
##
##    #--------------------------------------------------------------------------
##    # IOU of pred_box against boxes_all 
##    # confidence loss for objectness and non-objectness(background):
##    # 1. ignore_threshold = 0.5
##    # 2. non-objectness indicator : 1_noobj = (1.0 - true_obj)
##    #    bgd_conf = 1_noobj*(max_ious < ignore_threshold)
##    # 3. focal_loss : (true_obj - pred_obj)**2 *sigmoid_cross_entropy
##    ignore_threshold = 0.5
##    IOUS = utils.compute_iou(pred_box[:, :, :, :, tf.newaxis, :], boxes_all[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :])
##    max_iou = tf.expand_dims(tf.reduce_max(IOUS, axis=-1), axis=-1)
##    ignore_mask =  tf.cast( max_iou < ignore_threshold, tf.float32 )
##    bgd_conf = (1.0 - true_obj) * ignore_mask
##    cross_entropy_focal = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_obj, logits=conv_raw_obj)
##    focal_loss = tf.pow(true_obj- pred_obj, 2)* cross_entropy_focal
##
##    conf_loss = ( true_obj + bgd_conf )* focal_loss
##    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
##    #print('class_loss :', class_loss)
##    #print('iou_loss :', iou_loss)
##    #print('conf_loss :', conf_loss)
##    return [class_loss , xy_loss, wh_loss, conf_loss]
