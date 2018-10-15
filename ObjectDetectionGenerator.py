import glob
import cv2
import xml.etree.ElementTree as ET
from  keras.utils import Sequence
import numpy as np
from albumentations import (
    VerticalFlip, HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, Resize,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomSizedCrop
)

def PASCAL_VOC_bbox_2_yolo(bbox_orig, im_height, im_width, GRID_H = 13,  GRID_W = 13):
    """
    PASCAL VOC bounding box to YOLO bounding box. Returns CELL X and Y coordinates and normalized center and width, height
    """
    width = (bbox_orig[2] - bbox_orig[0])/im_width
    height = (bbox_orig[3] - bbox_orig[1])/im_height
    center_x = bbox_orig[0]/im_width + width/2
    center_y = bbox_orig[1]/im_height + height/2
    
    in_grid_H_units = center_y*GRID_H
    in_grid_W_units = center_x*GRID_W
    in_grid_H = int(in_grid_H_units)
    in_grid_W = int(in_grid_W_units)
    center_H = in_grid_H_units - in_grid_H
    center_W = in_grid_W_units - in_grid_W
    norm_H = np.abs(height*GRID_H)
    norm_W = np.abs(width*GRID_W)
    return (in_grid_H, in_grid_W), [center_W, center_H, norm_W, norm_H]

def annotation_to_nn_output(bboxes, category_ids, im_height, im_width, NUMBER_OF_CLASSES, NUMBER_OF_BBOXES, GRID_H, GRID_W, classes):
    """
    Recieves annotation dictionary and returns neural network output: [GRID_H, GRID_W, NUMBER_OF_BBOXES, 1 + NUMBER_OF_CLASSES + 4 ]
    """
    output = np.zeros([GRID_H, GRID_W, NUMBER_OF_BBOXES, 1 + NUMBER_OF_CLASSES + 4 ])
    cell_bboxes_count = {}
    for i, bbox in enumerate(bboxes):
        one_hot_class = np.zeros([len(classes)])
        one_hot_class[classes.index(category_ids[i])] = 1        
        (in_grid_H, in_grid_W), yolo_bbox = PASCAL_VOC_bbox_2_yolo(bbox, im_height, im_width, GRID_H,  GRID_W)
        if (in_grid_H, in_grid_W) not in cell_bboxes_count:
            cell_bboxes_count[(in_grid_H, in_grid_W)] = 0
        box_idx = cell_bboxes_count[(in_grid_H, in_grid_W)]
        output[in_grid_H, in_grid_W, box_idx, 0] = 1 # Object present
        output[in_grid_H, in_grid_W, box_idx, 1: 1 + NUMBER_OF_CLASSES] = one_hot_class # Object Class
        output[in_grid_H, in_grid_W, box_idx, 1 + NUMBER_OF_CLASSES:] = yolo_bbox # Bounding box
        cell_bboxes_count[(in_grid_H, in_grid_W)] = cell_bboxes_count[(in_grid_H, in_grid_W)] + 1
    return output

def get_PASCAL_VOC_classes(ANNOTATIONS_FOLDER):
    all_classes = []
    for filename in glob.glob(ANNOTATIONS_FOLDER+'/*'):
        all_classes.append(filename.split('/')[-1])
    return all_classes

def PASCAL_VOC_XML_to_dict(xml_file):
    annot_dict = {}
    xml_root = ET.parse(xml_file).getroot()
    annot_dict['filename'] = xml_root.find('filename').text
    annot_dict['folder'] = xml_root.find('folder').text
    width = int(xml_root.find('size').find('width').text)
    height = int(xml_root.find('size').find('height').text)
    depth = xml_root.find('size').find('depth').text
    annot_dict['size'] = {'width':width, 'height':height, 'depth':depth}
    
    objects = xml_root.findall('object')
    annot_dict['bboxes'] = []
    annot_dict['category_ids'] = []
    for obj in objects:
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        bbox_arr = [int(bbox.find('xmin').text), 
                       int(bbox.find('ymin').text),
                       int(bbox.find('xmax').text),
                       int(bbox.find('ymax').text)]
        annot_dict['bboxes'].append(bbox_arr)
        annot_dict['category_ids'].append(name)
    return annot_dict

def get_PASCAL_VOC_annotations_files(classes, folder):
    filenames = []
    for cl in classes:
        class_filenames = glob.glob(folder+'/'+cl+'/'+'*.xml')
        filenames = filenames + class_filenames
    return filenames

def get_image(annot_dic, folder, rescale):
    filename = folder + '/' + annot_dic['folder'] + '/' + annot_dic['filename'] + '.JPEG'
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img*rescale

def get_train_val_annotations_split(ANNOTATIONS_FOLDER, classes=None, MAX_NUMBER_OF_BBOXES_PER_CELL=1, split_ratio = 0.2, CLASSES_BY_DIR = True):
    if classes is None:
        classes = get_PASCAL_VOC_classes(ANNOTATIONS_FOLDER)
    NUMBER_OF_CLASSES = len(classes)
    annotations_train = []
    annotations_val = []
    if CLASSES_BY_DIR:
        for i, cl in enumerate(classes):
            annotations_class = []
            filenames = get_PASCAL_VOC_annotations_files([cl], ANNOTATIONS_FOLDER)
            for xml_file in filenames:
                annot_dict = PASCAL_VOC_XML_to_dict(xml_file)
                n_bboxes = len(annot_dict['bboxes'])
                if n_bboxes > 0 and n_bboxes <= MAX_NUMBER_OF_BBOXES_PER_CELL:
                    annotations_class.append(annot_dict)
            n_train = int(len(annotations_class)*(1-split_ratio))
            annotations_class = np.array(annotations_class)
            np.random.shuffle(annotations_class)
            annotations_train = annotations_train + list(annotations_class[:n_train])
            annotations_val = annotations_val + list(annotations_class[n_train:])
            print('({}/{}) Class {} has {} images in train and {} images in val'.format(i+1, NUMBER_OF_CLASSES, cl, len(annotations_class[:n_train]), len(annotations_class[n_train:])))
    else:
        filenames = glob.glob(ANNOTATIONS_FOLDER+'/'+'*.xml')
        n_files = len(filenames)
        classes_count = {class_id:0 for class_id in classes}
        for i, filename in enumerate(filenames):
            annot_dict = PASCAL_VOC_XML_to_dict(filename)
            n_bboxes = len(annot_dict['bboxes'])
            if n_bboxes > 0 and n_bboxes <= MAX_NUMBER_OF_BBOXES_PER_CELL:
                for cat_id in annot_dict['category_ids']:
                    if cat_id in classes:
                        classes_count[cat_id] = classes_count[cat_id] + 1
                        annotations_val.append(annot_dict)
            print('\rfile {}/{}'.format(i+1, n_files), end='')
        print()
        print(classes_count)
    print('found {} images in {} classes'.format(len(annotations_train)+len(annotations_val), NUMBER_OF_CLASSES))
    annotations_train = np.array(annotations_train)
    np.random.shuffle(annotations_train)
    annotations_val = np.array(annotations_val)
    np.random.shuffle(annotations_val)
    
    return annotations_train, annotations_val

def get_train_val_generators(ANNOTATIONS_FOLDER, IMAGES_FOLDER, batch_size, images_target_size, split_ratio = 0.2, classes=None, NUMBER_OF_BBOXES=1, GRID_H = 13, GRID_W = 13, rescale = 1.0/255.0):
    annotations_train, annotations_val = get_train_val_annotations_split(ANNOTATIONS_FOLDER, classes=classes, NUMBER_OF_BBOXES=NUMBER_OF_BBOXES, split_ratio = split_ratio)
    train_generator = ObjectDetectionGenerator(annotations_train, IMAGES_FOLDER, batch_size, images_target_size, classes, NUMBER_OF_BBOXES, GRID_H, GRID_W, rescale)
    val_generator = ObjectDetectionGenerator(annotations_val, IMAGES_FOLDER, batch_size, images_target_size, classes, NUMBER_OF_BBOXES, GRID_H, GRID_W, rescale)
    return train_generator, val_generator
    

class ObjectDetectionGenerator(Sequence):
    def __init__(self, annotations, IMAGES_FOLDER, batch_size, images_target_size, classes=None, NUMBER_OF_BBOXES = 1, GRID_H = 13, GRID_W = 13, rescale = 1.0/255.0):
        self.annotations = annotations
        self.IMAGES_FOLDER = IMAGES_FOLDER
        self.batch_size = batch_size
        self.batch_index = 0
        self.rescale = rescale
        self.NUMBER_OF_BBOXES = NUMBER_OF_BBOXES
        self.GRID_H = GRID_H
        self.GRID_W = GRID_W
        self.im_height = images_target_size[0]
        self.im_width = images_target_size[1]
        self.augmentator = Compose(
            [
                Resize(p=1, height=self.im_height, width=self.im_width),
                RandomSizedCrop(p=0.8, min_max_height=(3*self.im_height/4, self.im_height), height = self.im_height, width = self.im_width),
                HorizontalFlip(p=0.5), 
                VerticalFlip(p=0.5),
            ], 
              bbox_params={'format': 'pascal_voc', 'min_area': 50.0, 'min_visibility': 0.10, 'label_fields': ['category_id']})
        
        if classes is None:
            classes = get_PASCAL_VOC_classes(ANNOTATIONS_FOLDER)
            
        self.classes = classes
        self.NUMBER_OF_CLASSES = len(classes)
        
        
        self.samples = len(self.annotations)
        
    def __len__(self):
        return int(np.ceil(self.samples / float(self.batch_size)))
    
    def __getitem__(self, idx):
        annot_batch =  self.annotations[self.batch_index * self.batch_size: (self.batch_index + 1) * self.batch_size]
        self.batch_index = (self.batch_index + 1) % len(self)
        depth = 3
        images_array = np.zeros([len(annot_batch), self.im_height, self.im_width, depth])
        nn_output = np.zeros([len(annot_batch), self.GRID_H, self.GRID_W, self.NUMBER_OF_BBOXES, 1 + self.NUMBER_OF_CLASSES + 4 ])
        for i, annot_dic in enumerate(annot_batch):
            image = get_image(annot_dic, self.IMAGES_FOLDER, self.rescale)
            annotations = {'image': image, 
               'bboxes': annot_dic['bboxes'], 
               'category_id': annot_dic['category_ids']}
            augmented = self.augmentator(**annotations)
            
            nn_output[i] = annotation_to_nn_output(augmented['bboxes'],
                                                   augmented['category_id'],
                                                   self.im_height,
                                                   self.im_width,
                                                     self.NUMBER_OF_CLASSES, 
                                                     self.NUMBER_OF_BBOXES, 
                                                     self.GRID_H, 
                                                     self.GRID_W, 
                                                     self.classes)
            
            images_array[i] = augmented['image']
        return images_array, nn_output
        
    def __next__(self):
        return self.__getitem__(0)
    def __iter__(self):
        return self
    
### Visualization methods ###

def visualize_bbox(img, bbox, class_id, class_idx_to_name, thickness=2, bbox_type = 'PASCAL_VOC'):
    BOX_COLOR = (255, 0, 0)
    TEXT_COLOR = (255, 255, 255)
    if np.max(img) <= 1:
        BOX_COLOR = (1, 0, 0)
        TEXT_COLOR = (1, 1, 1)
    if bbox_type == 'PASCAL_VOC':
        x_min, y_min, x_max, y_max = np.array(bbox).astype(int)
    else:
        x_min, y_min, w, h = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
        
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=BOX_COLOR, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img

def yolo_bbox_2_PASCAL_VOC(grid_coordinates, bbox_yolo, im_height, im_width, GRID_H, GRID_W):
    step_y = im_height/GRID_H
    step_x = im_width/GRID_W
    bb_width = bbox_yolo[2]*step_x
    bb_height = bbox_yolo[3]*step_y
    bb_left = (grid_coordinates[1] + bbox_yolo[0])*step_x - bb_width/2
    bb_top = (grid_coordinates[0] + bbox_yolo[1])*step_y - bb_height/2
    return bb_left, bb_top, bb_left+bb_width, bb_top+bb_height

def annotate_image_from_nn_out(img, nn_out, classes, class_idx_to_name):
    img_with_annot = img.copy()
    obj_indexes = np.where(nn_out[:,:,:,0] == 1)
    im_height = img.shape[0]
    im_width = img.shape[1]
    for i, grid_out in enumerate(nn_out[obj_indexes]):
        class_id = classes[np.argmax(grid_out[1: 1 + len(class_idx_to_name)])]
        bbox = yolo_bbox_2_PASCAL_VOC((obj_indexes[0][i], obj_indexes[1][i]), 
                                      grid_out[1 + len(class_idx_to_name):], 
                                      im_height, im_width, 
                                      nn_out.shape[0], nn_out.shape[1])
        img_with_annot = visualize_bbox(img_with_annot, bbox, class_id, class_idx_to_name) 
    return img_with_annot