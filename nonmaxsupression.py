import torch
from IOU import intersection_over_union

def non_max_supression(bounding_boxes,iou_threshold,prob_threshold,box_format="corners"):
    assert type(bounding_boxes)==list
    #This is because our prediction is a list of bounding boxes
    bounding_boxes=[box for box in bounding_boxes if box[1]>prob_threshold]
    #For each bounding box there are 6 components . First is the class it belongs to,
    #second is its probability and rest 4 are x1,y1,x2,y2 .
    #So box[1] in our code refers to the probability threshold component
    bounding_boxes=sorted(bounding_boxes,key = lambda x:x[1],reverse=True)
    bounding_box_nms=[]

    while bounding_boxes:
        chosen_box=bounding_boxes.pop(0)
        #This is because the bounding box with highest probability is in first after sorting
        bounding_boxes = [
            box
            for box in bounding_boxes
            if box[0] != chosen_box[0]
            #This is to check if those two objects are of same class
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            <iou_threshold
        ]
        bounding_box_nms.append(chosen_box)

    return bounding_box_nms

