from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, GlobalAveragePooling2D, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
import tensorflow as tf
from keras import backend as K
from keras.applications.vgg16 import VGG16

# Arquitectures
def VGG16_for_YOLO_model(IMAGE_H, IMAGE_W, NUMBER_OF_CLASSES, NUMBER_OF_BBOXES, weights='imagenet', trainable_from_layer = 17, dropout_rate=0.5):
    input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
    modelVGG16 = VGG16(include_top=False, weights=weights)
    print('Model trainable from layer', modelVGG16.layers[trainable_from_layer].name)
    for layer in modelVGG16.layers[:trainable_from_layer]:
        layer.trainable = False
    for layer in modelVGG16.layers[trainable_from_layer:]:
        layer.trainable = True
    
    x = modelVGG16(input_image)
    x = BatchNormalization(name='norm_1')(x)
    x = Dropout(dropout_rate)(x)
    #x = Conv2D(512, kernel_size= (10,10), padding='same')(VGG16out)
    #x = Conv2D(512, kernel_size= (1,1), padding='same')(x)
    #x = Conv2D(512, kernel_size= (1,1), padding='same')(x)
    x = Conv2D(NUMBER_OF_BBOXES * (4 + 1 + NUMBER_OF_CLASSES), kernel_size= (1,1), padding='same')(x)
    
    GRID_H = x.shape[1].value
    GRID_W = x.shape[2].value
    output = Reshape((GRID_H, GRID_W, NUMBER_OF_BBOXES, 4 + 1 + NUMBER_OF_CLASSES))(x)
    
    model = Model(inputs=input_image, outputs=output)

    
    return model

def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)
    
def YOLO_V2_model(IMAGE_H, IMAGE_W, NUMBER_OF_CLASSES, NUMBER_OF_BBOXES, GAP=False):
    if IMAGE_H%32 != 0:
        print('IMAGE_H should be divisible by 32')
        return 
    if IMAGE_W%32 != 0:
        print('IMAGE_W should be divisible by 32')
        return
        
    input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))

    # Layer 1
    x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
    x = BatchNormalization(name='norm_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
    x = BatchNormalization(name='norm_3')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 4
    x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
    x = BatchNormalization(name='norm_4')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
    x = BatchNormalization(name='norm_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 7
    x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
    x = BatchNormalization(name='norm_7')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 8
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
    x = BatchNormalization(name='norm_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
    x = BatchNormalization(name='norm_9')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
    x = BatchNormalization(name='norm_10')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
    x = BatchNormalization(name='norm_11')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
    x = BatchNormalization(name='norm_12')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
    x = BatchNormalization(name='norm_13')(x)
    x = LeakyReLU(alpha=0.1)(x)

    skip_connection = x

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
    x = BatchNormalization(name='norm_14')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
    x = BatchNormalization(name='norm_15')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
    x = BatchNormalization(name='norm_16')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
    x = BatchNormalization(name='norm_17')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 18
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
    x = BatchNormalization(name='norm_18')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
    x = BatchNormalization(name='norm_19')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
    x = BatchNormalization(name='norm_20')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 21
    skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
    skip_connection = BatchNormalization(name='norm_21')(skip_connection)
    skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
    skip_connection = Lambda(space_to_depth_x2)(skip_connection)

    x = concatenate([skip_connection, x])

    # Layer 22
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
    x = BatchNormalization(name='norm_22')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 23
    x = Conv2D(NUMBER_OF_BBOXES * (4 + 1 + NUMBER_OF_CLASSES), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
    

    GRID_H = x.shape[1].value
    GRID_W = x.shape[2].value
    

    if GAP:
        output = GlobalAveragePooling2D(name='concatenated_outputs')(x)
        if NUMBER_OF_BBOXES>1:
            output = Reshape((NUMBER_OF_BBOXES, 4 + 1 + NUMBER_OF_CLASSES))(output)  
    else:
        output = Reshape((GRID_H, GRID_W, NUMBER_OF_BBOXES, 4 + 1 + NUMBER_OF_CLASSES))(x)

    model = Model([input_image], output)
    return model


class Metrics():
    def __init__(self, NUMBER_OF_CLASSES, start_classes_idx=1):
        self.NUMBER_OF_CLASSES = NUMBER_OF_CLASSES
        self.start_classes_idx = start_classes_idx
        self.start_bbox_idx = 1 + NUMBER_OF_CLASSES
        
    def classes_accuracy(self):
        def classes_accuracy(y_true, y_pred):
            indexes = tf.where(K.equal(y_true[:,:,:,:,0], K.ones_like(y_true[:,:,:,:,0])))
            y_true_pos = tf.gather_nd(y_true, indexes)[:, self.start_classes_idx:self.start_classes_idx+self.NUMBER_OF_CLASSES]
            y_pred_pos = tf.gather_nd(y_pred, indexes)[:, self.start_classes_idx:self.start_classes_idx+self.NUMBER_OF_CLASSES]
            return K.cast(K.equal(K.argmax(y_true_pos, axis=-1),
                                  K.argmax(y_pred_pos, axis=-1)),
                          K.floatx())
        return classes_accuracy
    
    def object_accuracy(self):
        def object_accuracy(y_true, y_pred):
            indexes = tf.where(K.equal(y_true[:,:,:,:,0], K.ones_like(y_true[:,:,:,:,0])))
            y_true_pos = tf.gather_nd(y_true, indexes)
            y_pred_pos = tf.gather_nd(y_pred, indexes)
            return K.mean(K.equal(y_true_pos[:,:1], K.round(K.sigmoid(y_pred_pos[:,:1]))), axis=-1)
        return object_accuracy
    
    def no_object_accuracy(self):
        def no_object_accuracy(y_true, y_pred):
            indexes_neg = tf.where(K.equal(y_true[:,:,:,:,0], K.zeros_like(y_true[:,:,:,:,0])))
            y_true_pos = tf.gather_nd(y_true, indexes_neg)
            y_pred_pos = tf.gather_nd(y_pred, indexes_neg)
            return K.mean(K.equal(y_true_pos[:,:1], K.round(K.sigmoid(y_pred_pos[:,:1]))), axis=-1)
        return no_object_accuracy
    
    def IOU(self):
        def IOU(y_true, y_pred):
            indexes = tf.where(K.equal(y_true[:,:,:,:,0], K.ones_like(y_true[:,:,:,:,0])))
            y_true_pos = tf.gather_nd(y_true, indexes)
            y_pred_pos = tf.gather_nd(y_pred, indexes)
            boxA = y_true_pos[:,self.start_bbox_idx:self.start_bbox_idx+4]
            boxB = y_pred_pos[:,self.start_bbox_idx:self.start_bbox_idx+4]
            xA = K.stack([boxA[:,0]-boxA[:,2]/2, boxB[:,0]-boxB[:,2]/2], axis=-1)
            yA = K.stack([boxA[:,1]-boxA[:,3]/2, boxB[:,1]-boxB[:,3]/2], axis=-1)
            xB = K.stack([boxA[:,0]+boxA[:,2]/2, boxB[:,0]+boxB[:,2]/2], axis=-1)
            yB = K.stack([boxA[:,1]+boxA[:,3]/2, boxB[:,1]+boxB[:,3]/2], axis=-1)

            xA = K.max(xA, axis=-1)
            yA = K.max(yA, axis=-1)
            xB = K.min(xB, axis=-1)
            yB = K.min(yB, axis=-1)

            interX = K.zeros_like(xB)
            interY = K.zeros_like(yB)

            interX = K.stack([interX, xB-xA], axis=-1)
            interY = K.stack([interY, yB-yA], axis=-1)

            #because of these "max", interArea may be constant 0, without gradients, and you may have problems with no gradients. 
            interX = K.max(interX, axis=-1)
            interY = K.max(interY, axis=-1)
            interArea = interX * interY

            boxAArea = (boxA[:,2]) * (boxA[:,3])    
            boxBArea = (boxB[:,2]) * (boxB[:,3]) 
            iou = interArea / (boxAArea + boxBArea - interArea)
            return iou
        return IOU

class Losses():
    def __init__(self, NUMBER_OF_CLASSES, start_classes_idx=1):
        self.NUMBER_OF_CLASSES = NUMBER_OF_CLASSES
        self.start_classes_idx = start_classes_idx
        self.start_bbox_idx = 1 + NUMBER_OF_CLASSES
    
    def bounding_box_mse(self):
        def bounding_box_mse(y_true, y_pred):
            indexes = tf.where(K.equal(y_true[:,:,:,:,0], K.ones_like(y_true[:,:,:,:,0])))
            y_true_pos = tf.gather_nd(y_true, indexes)[:,self.start_bbox_idx:self.start_bbox_idx+4]
            y_pred_pos = tf.gather_nd(y_pred, indexes)[:,self.start_bbox_idx:self.start_bbox_idx+4]
            return K.mean(K.square(y_pred_pos - y_true_pos), axis=-1)
        return bounding_box_mse
        
    def categorical_cross_entropy_loss(self):
        def categorical_cross_entropy_loss(y_true, y_pred):
            indexes = tf.where(K.equal(y_true[:,:,:,:,0], K.ones_like(y_true[:,:,:,:,0])))
            y_true_pos = tf.gather_nd(y_true, indexes)[:, self.start_classes_idx:self.start_classes_idx+self.NUMBER_OF_CLASSES]
            y_pred_pos = tf.gather_nd(y_pred, indexes)[:, self.start_classes_idx:self.start_classes_idx+self.NUMBER_OF_CLASSES]
            return K.categorical_crossentropy(y_true_pos, K.softmax(y_pred_pos))
        return categorical_cross_entropy_loss
    
    def object_bin_cross_entropy_loss(self):
        def object_bin_cross_entropy_loss(y_true, y_pred):
            indexes = tf.where(K.equal(y_true[:,:,:,:,0], K.ones_like(y_true[:,:,:,:,0])))
            y_true_pos = tf.gather_nd(y_true, indexes)
            y_pred_pos = tf.gather_nd(y_pred, indexes)
            return K.mean(K.binary_crossentropy(y_true_pos[:,:1], K.sigmoid(y_pred_pos[:,:1])), axis=-1)
        return object_bin_cross_entropy_loss
    
    def no_object_bin_cross_entropy_loss(self):
        def no_object_bin_cross_entropy_loss(y_true, y_pred):
            indexes_neg = tf.where(K.equal(y_true[:,:,:,:,0], K.zeros_like(y_true[:,:,:,:,0])))
            y_true_pos = tf.gather_nd(y_true, indexes_neg)
            y_pred_pos = tf.gather_nd(y_pred, indexes_neg)
            return K.mean(K.binary_crossentropy(y_true_pos[:,:1], K.sigmoid(y_pred_pos[:,:1])), axis=-1)
        return no_object_bin_cross_entropy_loss
    
    def focal_loss(self, gamma=2., alpha=.25):
        def focal_loss(y_true, y_pred):
            y_pred_sig = K.sigmoid(y_pred[:,:,:,:,0])
            pt_1 = tf.where(K.equal(y_true[:,:,:,:,0], K.ones_like(y_true[:,:,:,:,0])), y_pred_sig, tf.ones_like(y_pred_sig))
            pt_0 = tf.where(K.equal(y_true[:,:,:,:,0], K.zeros_like(y_true[:,:,:,:,0])), y_pred_sig, tf.zeros_like(y_pred_sig))              
            focal_loss = -(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1)) - \
                ((1-alpha) * K.pow( pt_0, gamma) * K.log(K.epsilon() + 1. - pt_0))
            return K.mean(focal_loss, axis = [-1, -2, -3])
        return focal_loss
    
    def YOLO_loss(self, k_classes = 1, k_bboxes = 1, k_no_object=0.5, k_object = 1):
        def YOLO_loss(y_true, y_pred):
            classes_cross_entropy = K.mean(self.categorical_cross_entropy_loss()(y_true, y_pred))
            bounding_box_mse = K.mean(self.bounding_box_mse()(y_true, y_pred))
            object_loss = K.mean(self.object_bin_cross_entropy_loss()(y_true, y_pred))
            no_object_loss = K.mean(self.no_object_bin_cross_entropy_loss()(y_true, y_pred))
            return k_classes*classes_cross_entropy + k_bboxes*bounding_box_mse + k_no_object*no_object_loss + k_object*object_loss
        return YOLO_loss
   
    def YOLO_loss_focal_loss(self, k_classes = 1, k_bboxes = 1, gamma=2., alpha=.25):
        def YOLO_loss_focal_loss(y_true, y_pred):
            classes_cross_entropy = K.mean(self.categorical_cross_entropy_loss()(y_true, y_pred))
            bounding_box_mse = K.mean(self.bounding_box_mse()(y_true, y_pred))
            focal_loss = K.mean(self.focal_loss(gamma, alpha)(y_true, y_pred))
            return k_classes*classes_cross_entropy + k_bboxes*bounding_box_mse + focal_loss
        return YOLO_loss_focal_loss

class ObjectDectection():
    def __init__(self, CLASSES_ID_TO_NAME, GRID_H, GRID_W, NUMBER_OF_BBOXES=1, ARQUITECTURE='YOLO_V2', weights = 'imagenet', trainable_from_layer=17, dropout_rate = 0.5, CELL_SIZE_IN_PIXELS_H = 32, CELL_SIZE_IN_PIXELS_W = 32):
        self.CLASSES_ID_TO_NAME = CLASSES_ID_TO_NAME
        self.CLASSES = list(self.CLASSES_ID_TO_NAME.keys())
        self.NUMBER_OF_CLASSES = len(CLASSES_ID_TO_NAME)
        self.GRID_H = GRID_H
        self.GRID_W = GRID_W
        self.IMAGE_H = CELL_SIZE_IN_PIXELS_H*GRID_H
        self.IMAGE_W = CELL_SIZE_IN_PIXELS_W*GRID_W
        self.NUMBER_OF_BBOXES = NUMBER_OF_BBOXES
        if ARQUITECTURE == 'YOLO_V2':
            self.model = YOLO_V2_model(self.IMAGE_H, self.IMAGE_W, self.NUMBER_OF_CLASSES, NUMBER_OF_BBOXES)
        elif ARQUITECTURE == 'VGG16':
            self.model = VGG16_for_YOLO_model(self.IMAGE_H, self.IMAGE_W, self.NUMBER_OF_CLASSES, NUMBER_OF_BBOXES, weights, trainable_from_layer, dropout_rate)
        self.metrics = Metrics(self.NUMBER_OF_CLASSES)
        self.losses = Losses(self.NUMBER_OF_CLASSES)
        
    
    
    
    
from matplotlib import pyplot as plt
from IPython.display import clear_output

import matplotlib.patches as patches
import keras
import numpy as np
### PLOTING ###
class PlotLosses(keras.callbacks.Callback):
    def __init__(self, plot_interval=1):
        self.plot_interval = plot_interval
    
    def on_train_begin(self, logs={}):
        print('Begin training')
        self.i = 0
        self.x = []
        
        self.log_plots = {}
        self.total_losses = {}
        self.acc = {}
        self.abserrors = {}
        self.confidences = {}
        self.cat_output_loss = {}
        self.bb_loss = {}
        self.ious = {}
        self.logs = []
        
    def on_epoch_end(self, epoch, logs={}):
        if len(self.log_plots) == 0:
            for k,v in logs.items():
                name = k.replace('val_','')
                if name not in self.log_plots:
                    self.log_plots[name] = {}
                    self.log_plots[name][name] = []
                    if 'val_'+name in logs:
                        self.log_plots[name]['val_'+name] = []
                    
        for i, (name, ne) in enumerate(self.log_plots.items()):
            for k,v in ne.items():
                if k in logs:
                    self.log_plots[name][k].append(logs[k])
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.i += 1
        if (epoch%self.plot_interval==0):
            clear_output(wait=True)
            subplots = len(self.log_plots)
            rows = int(np.ceil(subplots / 2))
            
            f, axs = plt.subplots(rows, 2, sharex=True, figsize=(20,5*rows))
            axs = axs.flatten()
            for i, (name, ne) in enumerate(self.log_plots.items()):
                for k,v in ne.items():
                    if 'val' in k:
                        axs[i].plot(self.x, v, label=k, ls='-.', color='b')
                    else:
                        axs[i].plot(self.x, v, label=k, color='r')        
                    
                axs[i].legend()
                axs[i].grid()
            plt.show()