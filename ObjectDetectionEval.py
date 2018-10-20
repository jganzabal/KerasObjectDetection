import numpy as np

def compare_with_annot(last_pred, last_annot, iou_thres = 0.5):
    """
    Return FP, FN, TP for each class in the image
    """
    classes_results = {}
    for class_idx, pred_dict in last_pred.items():
        classes_results[class_idx] = {}
        classes_results[class_idx]['N'] = len(last_annot[class_idx]['bboxes'])
        hits = 0
        for bbox_annot in last_annot[class_idx]['bboxes']:
            for bbox_pred in pred_dict['bboxes']:
                iou, _ = getIUO(bbox_annot.reshape(1, 4), bbox_pred.reshape(1, 4))
                if iou>iou_thres:
                    hits = hits + 1
        total_objs = len(last_annot[class_idx]['bboxes'])
        total_pred_objs = len(last_pred[class_idx]['bboxes'])
        TP = np.min([hits, total_objs])
        classes_results[class_idx]['TP'] = TP
        FP = (total_pred_objs - TP)
        classes_results[class_idx]['FP'] = FP * (FP>0)
        classes_results[class_idx]['FN'] = (total_objs - TP) * (total_objs > hits)
    return classes_results

def get_IOUs_enhanced(annotations, predictions, n_classes, consider_class = True):
    """
        Selects the best bounding box out of NUMBER_OF_BBOX possible bboxes and calculates the average IOU
        if consider_class is true then predicted_class has to be equal to annotated class to consider de IOU
    """
    NUMBER_OF_BBOX = annotations.shape[-2]
    obj_indexes = np.where(annotations[:,:,:,:,0] == 1)
    annotated_bboxes = annotations[obj_indexes][:][:,n_classes+1:n_classes+1+4]
    bboxes_iou = np.zeros([annotated_bboxes.shape[0], NUMBER_OF_BBOX])
    annotated_class = np.argmax(annotations[obj_indexes][:][:, 1: 1 + n_classes], axis = -1)
    bboxes_indexes = (*obj_indexes[0:-1],)
    predictions_filtered = predictions[bboxes_indexes]
    for i in range(NUMBER_OF_BBOX):
        predicted_data = predictions_filtered[:,i,:]
        predicted_bboxes = predicted_data[:,n_classes+1:n_classes+1+4]
        predicted_class = np.argmax(predicted_data[:, 1: 1 + n_classes], axis = -1)
        predicted_class_prob = np.max(softmax(predicted_data[:, 1: 1 + n_classes]), axis = -1)
        predicted_obj_prob = sigmoid(predicted_data[:, 0])
        IOUs, _ = getIUO(annotated_bboxes, 
                         predicted_bboxes, 
                         from_center_to_box=True)
        if consider_class:
            bboxes_iou[:,i] = IOUs  * (predicted_class == annotated_class) # * predicted_obj_prob * predicted_class_prob
        else:
            bboxes_iou[:,i] = IOUs
    best_bbox_idxs = np.argmax(bboxes_iou, axis = 1)
    best_bbox_ious = np.max(bboxes_iou, axis = 1)
    return np.mean(best_bbox_ious)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x.T - np.max(x, axis = -1))
    return (e_x / e_x.sum(axis=0)).T

def getBB_area(bb):
    IntersectionArea = (bb[:,2] - bb[:,0])*(bb[:,3] - bb[:,1])
    return IntersectionArea

def getIUO(bb1, bb2, from_center_to_box = False):
    if from_center_to_box:
        bb1 = np.array([
             bb1[:,0] - bb1[:,2]/2, 
             bb1[:,1] - bb1[:,3]/2,
             bb1[:,0] + bb1[:,2]/2, 
             bb1[:,1] + bb1[:,3]/2,]).T
        bb2 = np.array([
             bb2[:,0] - bb2[:,2]/2, 
             bb2[:,1] - bb2[:,3]/2,
             bb2[:,0] + bb2[:,2]/2, 
             bb2[:,1] + bb2[:,3]/2,]).T

    intersection_bb = np.array([np.vstack([bb1[:,0], bb2[:,0]]).max(axis=0),
        np.vstack([bb1[:,1], bb2[:,1]]).max(axis=0),
        np.vstack([bb1[:,2], bb2[:,2]]).min(axis=0),
        np.vstack([bb1[:,3], bb2[:,3]]).min(axis=0)]).T
    no_intersec = 1*(intersection_bb[:,3]-intersection_bb[:,1]>0)*(intersection_bb[:,2]-intersection_bb[:,0]>0)
    intersection_bb = (intersection_bb.T * no_intersec).T
    IntersectionArea = no_intersec*getBB_area(intersection_bb)
    
    areaBB1 = getBB_area(bb1)
    areaBB2 = getBB_area(bb2)
    denom = areaBB1 + areaBB2 - IntersectionArea
    # When both bbox are 0,0,0,0
    no_data_idxes = np.where(denom == 0)
    denom[no_data_idxes] = 1
    
    IOU = IntersectionArea/denom
    return IOU, intersection_bb

def yolo_bbox_2_PASCAL_VOC(grid_coordinates, bbox_yolo, im_height, im_width, GRID_H, GRID_W):
    step_y = im_height/GRID_H
    step_x = im_width/GRID_W
    bb_width = bbox_yolo[2]*step_x
    bb_height = bbox_yolo[3]*step_y
    bb_left = (grid_coordinates[1] + bbox_yolo[0])*step_x - bb_width/2
    bb_top = (grid_coordinates[0] + bbox_yolo[1])*step_y - bb_height/2
    return bb_left, bb_top, bb_left+bb_width, bb_top+bb_height

def get_predictions_before_NMS(prediction, classes, im_width, im_height, thres = 0.5, activations = False):
    """
    
    """
    n_classes = len(classes)
    GRID_H = prediction.shape[0]
    GRID_W = prediction.shape[1]
    if activations:
        pred_idx_all = np.where(prediction[:,:,:, 0] >= thres)
    else:
        pred_idx_all = np.where(sigmoid(prediction[:,:,:, 0]) >= thres)
    pred_processed = {class_id:{'bboxes':[], 'joint_probs':[], 'class_probs':[], 'obj_probs':[]} for class_id in classes}
    for idx, bbox_index in enumerate(pred_idx_all[2]):
        cell_y = np.array([pred_idx_all[0][idx]]) # y cell position for bounding box 
        cell_x = np.array([pred_idx_all[1][idx]]) # x cell position for bounding box 
        pred_idx = (cell_y, cell_x, bbox_index)
        cell_prediction = prediction[pred_idx][0]
        pred_class_idx = np.argmax(cell_prediction[1:1+n_classes])
        predicted_class = classes[pred_class_idx]
        if activations:
            object_probability = cell_prediction[0]
            class_probability = max(cell_prediction[1: 1+n_classes])
        else:
            object_probability = sigmoid(cell_prediction[0])
            class_probability = max(softmax(cell_prediction[1: 1+n_classes]))
        
        if (class_probability*object_probability>=thres):
            predicted_yolo_box = cell_prediction[1+n_classes:]
            predicted_box = yolo_bbox_2_PASCAL_VOC((pred_idx[0][0], pred_idx[1][0]), 
                                          predicted_yolo_box, 
                                          im_height, im_width, GRID_H, GRID_W)

            pred_processed[predicted_class]['bboxes'].append(np.array(predicted_box))
            pred_processed[predicted_class]['class_probs'].append(class_probability)
            pred_processed[predicted_class]['obj_probs'].append(object_probability)
            pred_processed[predicted_class]['joint_probs'].append(object_probability*class_probability)
    return pred_processed



def non_max_supression(bboxes, probab, IOU_thres = 0.5):
    def get_remaining_boxes(bboxes, sel_box, probab, IOU_thres = IOU_thres):
        new_boxes = []
        new_probab = []
        for i, bbox in enumerate(bboxes):
            IOU, _ = getIUO(sel_box.reshape(1,4), bbox.reshape(1,4))
            # if there is low intersection, it can be another object
            if IOU<IOU_thres:
                new_boxes.append(bbox)
                new_probab.append(probab[i])
        return new_boxes, new_probab
    final_boxes = []
    final_probs = []
    while len(bboxes)>0:
        # Select most probable Bounding Box
        max_sel_idx = np.argmax(probab)
        # Append it to final boxes
        final_boxes.append(bboxes[max_sel_idx])
        final_probs.append(probab[max_sel_idx])
        bboxes, probab = get_remaining_boxes(bboxes, bboxes[max_sel_idx], probab)
    return final_boxes, final_probs

def get_predictions(prediction, classes, IMAGE_W, IMAGE_H, obj_conf_thres = 0.5, IOU_thres = 0.5, activations = False):
    bbox_predictions = get_predictions_before_NMS(prediction, classes, IMAGE_W, IMAGE_H, thres = obj_conf_thres, activations = activations)
    results = {}
    for class_id in classes:
        bboxes, probs = non_max_supression(bbox_predictions[class_id]['bboxes'], bbox_predictions[class_id]['joint_probs'], IOU_thres = IOU_thres)
        results[class_id] = {'bboxes':bboxes, 'probs': probs}
    return results
