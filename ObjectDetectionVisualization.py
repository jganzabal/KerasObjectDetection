import numpy as np
import cv2

def add_bboxes_to_image(image, results, class_idx_to_name, comp_res = None):
    image = image.copy()
    for class_id, v in results.items():
        #if comp_res is not None:
        #    print(comp_res[class_id])
        for i,predicted_box in enumerate(v['bboxes']):
            add_bbox_to_image(image, 
                              predicted_box, 
                              class_id , 
                              class_idx_to_name, 
                              thickness=int(5*v['probs'][i]), bbox_type = 'PASCAL_VOC',
                              text=' ' + str(i)
                             )
    return image

def add_bbox_to_image(img, bbox, class_id, class_idx_to_name, thickness=2, bbox_type = 'PASCAL_VOC', text = ''):
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
    class_name = class_idx_to_name[class_id] + text
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

def annotate_image_from_annotation(img, annotation, classes, class_idx_to_name):
    img_with_annot = img.copy()
    im_height = img.shape[0]
    im_width = img.shape[1]
    for i, bbox in enumerate(annotation['bboxes']):
        class_id = annotation['category_ids'][i]
        img_with_annot = visualize_bbox(img_with_annot, bbox, class_id, class_idx_to_name) 
    return img_with_annot