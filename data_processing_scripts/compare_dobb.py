import os
import argparse
import cv2
import numpy as np
import math
import utils
import pandas as pd
from utils import get_distance

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--labels_ref', default='',
                        help='the directory to the  reference labels')
    parser.add_argument('--labels_pred', default='',
                        help='the directory to the labels to be preded')
    parser.add_argument('--images', default='',
                        help='the directory to the images')
    parser.add_argument('--plots', default=None,
                        help='the directory where to save plots')
    parser.add_argument('--results', default=None,
                    help='where to save the results')
    parser.add_argument('--name', default="results",
                    help='name of the txt')
    parser.add_argument('--iou', default=0.5, type=float,
                    help='min IoU to match')
    parser.add_argument('--ellipse', default=False, action='store_true',
                    help='rectangle or ellipse?')
    parser.add_argument('--thickness', default=3, type=int,
                    help='line thickness?')
    parser.add_argument('--no_cls_match', default=False, action='store_true',
                    help='with this the class is not taken into account at matching')
    args = parser.parse_args()
    return args

def raw_labels(Labels, H, W):
    label_array = []
    for line in Labels:
        current_line = line[:-1]
        elements=current_line.split(" ")
        for i in range(len(elements)):
            if i==0:
                elements[i]=int(elements[i])
            elif i<9:
                if i%2:
                    elements[i]=int(float(elements[i])*W)
                else:
                    elements[i]=int(float(elements[i])*H)
        cosD=float(elements[9])
        sinD=float(elements[10])
        angleD=math.atan2(sinD,cosD)
        if angleD<0:
            angleD=math.pi*2+angleD
        rect = np.array((elements[1:9]))
        rect = cv2.minAreaRect(rect.reshape(4, 2))
        Dx,Dy=utils.dptFromAngle(rect[0][0],rect[0][1],rect[1][0],rect[1][1],math.radians(rect[2]),angleD)
        elements[9]=Dx
        elements[10]=Dy
        label_array.append(elements)
    return label_array

def match(labels_R, labels_P,ar):
    matched_labels_ref = []
    matched_labels_pred = []
    matched_ious = []
    IOUs = np.zeros((len(labels_R),len(labels_P)))
    unmatched_labels_ref = []
    unmatched_labels_pred = []
    for Ri in range(len(labels_R)):
        this_label_ref = labels_R[Ri]
        rect_ref = np.array((this_label_ref[1:9]))
        rect_ref_xywhr = cv2.minAreaRect(rect_ref.reshape(4, 2))        
        for Pi in range(len(labels_P)):
            if this_label_ref[0]==labels_P[Pi][0] or ar.no_cls_match:
                potential_label_pred=labels_P[Pi]
                rect_pred = np.array((potential_label_pred[1:9]))
                rect_pred_xywhr = cv2.minAreaRect(rect_pred.reshape(4, 2))
                r1 = cv2.rotatedRectangleIntersection(rect_ref_xywhr, rect_pred_xywhr)
                if r1[0] != 0:
                    rect_ref_xywhr_ar = np.array((rect_ref_xywhr[0][0],rect_ref_xywhr[0][1],rect_ref_xywhr[1][0],rect_ref_xywhr[1][1],rect_ref_xywhr[2]))
                    rect_pred_xywhr = np.array((rect_pred_xywhr[0][0],rect_pred_xywhr[0][1],rect_pred_xywhr[1][0],rect_pred_xywhr[1][1],rect_pred_xywhr[2]))
                    rect_ref_xywhr_ar[4] = math.radians(rect_ref_xywhr_ar[4])
                    rect_pred_xywhr[4] = math.radians(rect_pred_xywhr[4])
                    iou = utils.probiou(rect_ref_xywhr_ar,rect_pred_xywhr)
                    if iou>=ar.iou:
                        IOUs[Ri,Pi]=iou
    correctP = np.zeros(len(labels_P)).astype(bool)
    correctR = np.zeros(len(labels_R)).astype(bool)
    matches = np.nonzero(IOUs >= ar.iou)  # IoU > threshold and classes match
    matches = np.array(matches).T
    if matches.shape[0]:
        if matches.shape[0] > 1:
            matches = matches[IOUs[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correctR[matches[:, 0].astype(int)] = True
        correctP[matches[:, 1].astype(int)] = True
    for i in range(len(matches)):
        matched_labels_ref.append(labels_R[matches[i,0]])
        matched_labels_pred.append(labels_P[matches[i,1]])
    for i in range(len(correctR)):
        if not correctR[i]:
            unmatched_labels_ref.append(labels_R[i])
    for i in range(len(correctP)):
        if not correctP[i]:
            unmatched_labels_pred.append(labels_P[i])
    for i in range(len(matches)):
        matched_ious.append(IOUs[matches[i,0],matches[i,1]])
    return matched_labels_ref, matched_labels_pred, matched_ious, unmatched_labels_ref, unmatched_labels_pred

def get_angles(elements):
    x1 = elements[1]
    y1 = elements[2]
    x2 = elements[3]
    y2 = elements[4]
    x3 = elements[5]
    y3 = elements[6]
    x4 = elements[7]
    y4 = elements[8]
    cx,cy = utils.get_center(elements,integer=True)
    angle1 = math.atan2(y1-cy,x1-cx)
    angle2 = math.atan2(y2-cy,x2-cx)
    angle3 = math.atan2(y3-cy,x3-cx)
    angle4 = math.atan2(y4-cy,x4-cx)
    if angle1<0:
        angle1=2*math.pi+angle1
    if angle2<0:
        angle2=2*math.pi+angle2
    if angle3<0:
        angle3=2*math.pi+angle3
    if angle4<0:
        angle4=2*math.pi+angle4
    return angle1,angle2,angle3,angle4

def get_edge(elements):
    ### A-B: Edge1, B-C: Edge2, C-D: Edge3, D-A: Edge4
    cx,cy = utils.get_center(elements,integer=True)
    angle1,angle2,angle3,angle4 = get_angles(elements)
    kx = elements[9]
    ky = elements[10]
    angle_k = math.atan2(ky-cy,kx-cx)

    ## Found the special edge, where the angles change from 2Pi to 0
    if min(angle1,angle2)==min(angle1,angle2,angle3,angle4) and max(angle1,angle2)==max(angle1,angle2,angle3,angle4):
        spec=1
    elif min(angle2,angle3)==min(angle1,angle2,angle3,angle4) and max(angle2,angle3)==max(angle1,angle2,angle3,angle4):
        spec=2
    elif min(angle3,angle4)==min(angle1,angle2,angle3,angle4) and max(angle3,angle4)==max(angle1,angle2,angle3,angle4):
        spec=3
    elif min(angle4,angle1)==min(angle1,angle2,angle3,angle4) and max(angle4,angle1)==max(angle1,angle2,angle3,angle4):
        spec=4
    else:
        print("special edge not found, default")
        print(elements)
        print(angle1,angle2,angle3,angle4)
        print('')
        return 1,angle_k
    
    if angle_k<0:
        angle_k=2*math.pi+angle_k
    if ((angle_k<=min(angle1,angle2) or angle_k>=max(angle1,angle2)) and spec == 1) or ((angle_k>=min(angle1,angle2) and angle_k<=max(angle1,angle2)) and spec != 1):
        return 1,angle_k
    if ((angle_k<=min(angle2,angle3) or angle_k>=max(angle2,angle3)) and spec == 2) or ((angle_k>=min(angle2,angle3) and angle_k<=max(angle2,angle3)) and spec != 2):
        return 2,angle_k
    if ((angle_k<=min(angle3,angle4) or angle_k>=max(angle3,angle4)) and spec == 3) or ((angle_k>=min(angle3,angle4) and angle_k<=max(angle3,angle4)) and spec != 3):
        return 3,angle_k
    if ((angle_k<=min(angle4,angle1) or angle_k>=max(angle4,angle1)) and spec == 4) or ((angle_k>=min(angle4,angle1) and angle_k<=max(angle4,angle1)) and spec != 4):
        return 4,angle_k
    print("Error in the edge")
    print(elements)
    print(angle1,angle2,angle3,angle4)
    print(spec)
    print('')
    return None,None

if __name__ == '__main__':
    args = parse_args()
    labels_ref=[]
    dlist=os.listdir(args.labels_ref)
    dlist.sort()
    for filename in dlist:
        if filename.endswith(".txt") and not filename.startswith("classes"):
            #print(os.path.join(directory, filename))
            labels_ref.append(filename)
        else:
            continue
    if len(labels_ref)<1:
        print("%s is empty"%(args.labels_ref))
        exit()
    else:
        print("Checking %d files"%(len(labels_ref)))
    if args.plots is not None and not os.path.exists(args.plots):
        os.makedirs(args.plots)
    total_matched = 0
    total_matched_edges = 0
    total_number_refs = 0
    total_number_preds = 0
    total_angle_diffs = []
    total_angle_diffs_obb = []
    total_angle_diffs2 = []
    total_center_dist = []
    total_edge_dist = []
    total_class_match = 0
    total_ious = []
    total_FP = 0
    total_FN = 0

    if args.results is None:
        path_results = os.path.join(args.labels_ref, "results")
    else:
        path_results = args.results
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    f_results_detections=open(os.path.join(path_results, args.name+"_detections.txt"), "w")
    f_results_angle=open(os.path.join(path_results, args.name+"_angle.txt"), "w")
    f_results_center=open(os.path.join(path_results, args.name+"_center.txt"), "w")
    f_results_edge=open(os.path.join(path_results, args.name+"_edge.txt"), "w")
    f_results_iou=open(os.path.join(path_results, args.name+"_iou.txt"), "w")
    f_results_directions=open(os.path.join(path_results, args.name+"_directions.txt"), "w")
    for i in range(len(labels_ref)):
        label_name=labels_ref[i]
        print("Processing: "+label_name)
        img_extensions=["png","jpg","tif"]
        img = None
        for img_ext in img_extensions:
            if os.path.isfile(os.path.join(args.images, label_name[:-3]+img_ext)):
                img_name=label_name[:-3]+img_ext
                img = cv2.imread(os.path.join(args.images, img_name))
        if img is None:
            print("Image %s does not exists"%(label_name[:-3]))
            continue
        H,W,_ = img.shape

        f_label = open(os.path.join(args.labels_ref, label_name), "r")
        Lines_ref = f_label.readlines()
        f_label.close()
        labels_R = raw_labels(Lines_ref,H,W)

        f_label = open(os.path.join(args.labels_pred, label_name), "r")
        Lines_pred = f_label.readlines()
        f_label.close()
        labels_P = raw_labels(Lines_pred,H,W)
        
        matched_labels_ref, matched_labels_pred, matched_ious, unmatched_labels_ref, unmatched_labels_pred = match(labels_R,labels_P,args)
        
        total_FP+=len(unmatched_labels_pred)
        total_FN+=len(unmatched_labels_ref)
        local_number_refs=len(Lines_ref)
        total_number_refs+=len(Lines_ref)
        local_number_preds=len(Lines_pred)
        total_number_preds+=len(Lines_pred)

        total_matched+=len(matched_labels_ref)
        local_matched=len(matched_labels_ref)
        local_ious = matched_ious
        if local_matched>0:
            total_ious.append(matched_ious)

        local_angle_diffs = []
        local_center_dist = []
        local_edge_dist = []
        local_matched_edges = 0
        local_class_match = 0
        local_direction_ref = np.zeros((8))
        local_direction_pred = np.zeros((8))
        local_direction_diff = np.zeros((8))
        if args.ellipse:
            if len(unmatched_labels_ref)>0:
                img = utils.draw_ellipse(img,unmatched_labels_ref,colors=['red','red'],thickness=args.thickness,plot_kp="arrow")
            if len(unmatched_labels_pred)>0:
                img = utils.draw_ellipse(img,unmatched_labels_pred,colors=['orange','orange'],thickness=args.thickness,plot_kp="arrow")
        else:
            if len(unmatched_labels_ref)>0:
                img = utils.draw_rectangle(img,unmatched_labels_ref,colors=['red','red','red','red','red'],thickness=args.thickness,plot_kp="arrow")
            if len(unmatched_labels_pred)>0:
                img = utils.draw_rectangle(img,unmatched_labels_pred,colors=['orange','orange','orange','orange','orange'],thickness=args.thickness,plot_kp="arrow")

        for lindex in range(len(matched_labels_ref)):
            this_label_ref=matched_labels_ref[lindex]
            this_label_pred=matched_labels_pred[lindex]

            center_refX, center_refY = utils.get_center(this_label_ref,integer=True)
            center_predX, center_predY = utils.get_center(this_label_pred,integer=True)
            center_distance = get_distance(center_refX,center_refY,center_predX,center_predY)
            total_center_dist.append(center_distance)
            local_center_dist.append(center_distance)

            direction_refX = this_label_ref[9]
            direction_refY = this_label_ref[10]
            direction_predX = this_label_pred[9]
            direction_predY = this_label_pred[10]
            direction_obb_predX = (this_label_pred[1]+this_label_pred[3])/2
            direction_obb_predY = (this_label_pred[2]+this_label_pred[4])/2

            angle_ref_refdirection = math.atan2(direction_refY-center_refY,direction_refX-center_refX)
            angle_ref_preddirection = math.atan2(direction_predY-center_refY,direction_predX-center_refX)
            angle_diff2 = abs(math.atan2(math.sin(angle_ref_refdirection - angle_ref_preddirection),math.cos(angle_ref_refdirection - angle_ref_preddirection)))
            total_angle_diffs2.append(angle_diff2)

            angle_obb_preddirection = math.atan2(direction_obb_predY-center_refY,direction_obb_predX-center_refX)
            angle_diff_obb = abs(math.atan2(math.sin(angle_ref_refdirection - angle_obb_preddirection),math.cos(angle_ref_refdirection - angle_obb_preddirection)))
            total_angle_diffs_obb.append(angle_diff_obb)

            if this_label_ref[0]==this_label_pred[0]:
                local_class_match+=1
                total_class_match+=1

            ### A-B: Edge1, B-C: Edge2, C-D: Edge3, D-A: Edge4
            this_label_pred = utils.order_elements(this_label_ref,this_label_pred)
            edge_ref, angle_ref = get_edge(this_label_ref)
            edge_pred, angle_pred = get_edge(this_label_pred)
            direction_ref = int(round(math.degrees(angle_ref)/45))
            if direction_ref==8:
                direction_ref=0
            direction_pred = int(round(math.degrees(angle_pred)/45))
            if direction_pred==8:
                direction_pred=0
            local_direction_ref[direction_ref]+=1
            local_direction_pred[direction_pred]+=1
            edge_diff = 0 ## 0 - 0, 1 - 90, 2 - 180
            cls_ref = this_label_ref[0]
            cls_pred = this_label_pred[0]
            if edge_pred==edge_ref:
                local_matched_edges+=1
                total_matched_edges+=1
            else:
                edge_diff = abs(edge_ref-edge_pred)
                if edge_diff==3:
                    edge_diff = 1
            total_edge_dist.append(edge_diff)
            local_edge_dist.append(edge_diff)
            total_edge_dist.append(cls_ref)
            local_edge_dist.append(cls_ref)
            total_edge_dist.append(cls_pred)
            local_edge_dist.append(cls_pred)
            
            angle_diff = abs(math.atan2(math.sin(angle_ref - angle_pred),math.cos(angle_ref - angle_pred)))         
            total_angle_diffs.append(angle_diff)

            
            local_angle_diffs.append(angle_diff)

            if args.ellipse:
                img = utils.draw_ellipse(img,this_label_ref,colors=['yellow','yellow'],thickness=args.thickness,plot_kp="arrow")
                img = utils.draw_ellipse(img,this_label_pred,colors=['black','black'],thickness=args.thickness+4,plot_kp="arrow")
                img = utils.draw_ellipse(img,this_label_pred,colors=['green','green'],thickness=args.thickness,plot_kp="arrow")
            else:
                img = utils.draw_rectangle(img,this_label_ref,colors=['yellow','yellow','yellow','yellow','yellow'],thickness=args.thickness,plot_kp="arrow")
                img = utils.draw_rectangle(img,this_label_pred,colors=['green','green','green','green','green'],thickness=args.thickness,plot_kp="arrow")
        
        cv2.imwrite(os.path.join(args.plots,img_name),img)
        f_results_detections.write("%d %d %d %d %d\n"%(local_number_refs,local_number_preds,local_matched,local_matched_edges,local_class_match))
        if local_matched>0:
            for elementI in range(len(local_angle_diffs)-1):
                f_results_angle.write("%f "%(local_angle_diffs[elementI]))
            f_results_angle.write("%f\n"%(local_angle_diffs[len(local_angle_diffs)-1]))
            for elementI in range(len(local_center_dist)-1):
                f_results_center.write("%f "%(local_center_dist[elementI]))
            f_results_center.write("%f\n"%(local_center_dist[len(local_center_dist)-1]))
            for elementI in range(len(local_edge_dist)-1):
                f_results_edge.write("%d "%(local_edge_dist[elementI]))
            f_results_edge.write("%d\n"%(local_edge_dist[len(local_edge_dist)-1]))
            for elementI in range(len(local_ious)-1):
                f_results_iou.write("%f "%(local_ious[elementI]))
            f_results_iou.write("%f\n"%(local_ious[len(local_ious)-1]))
            for elementI in range(len(local_direction_ref)-1):
                f_results_directions.write("%d %d %d "%(local_direction_ref[elementI],local_direction_pred[elementI],local_direction_diff[elementI]))
            f_results_directions.write("%d %d %d\n"%(local_direction_ref[len(local_direction_ref)-1],local_direction_pred[len(local_direction_ref)-1],local_direction_diff[len(local_direction_ref)-1]))

    f_results_detections.close()
    f_results_angle.close()
    f_results_center.close()
    f_results_edge.close()
    f_results_iou.close()
    f_results_directions.close()

    df = pd.DataFrame(total_angle_diffs)
    df.to_csv(os.path.join(path_results, args.name+"_angle.csv"),header=False, index=False)

    print("\nSummary\n")

    print("Number of refs: %d"%(total_number_refs))
    print("Number of preds: %d"%(total_number_preds))
    print("Matches: %d (%.2f%% of GT labels) (%.2f%% of Pred labels)"%(total_matched,(total_matched/total_number_refs)*100,(total_matched/total_number_preds)*100))
    print("F-score")
    print("TP: %d, FN: %d, FP: %d"%(total_matched, total_FN,total_FP))
    print("F1: %f"%((2*total_matched)/(2*total_matched+total_FN+total_FP)))
    prec=total_matched/(total_matched+total_FP)
    rec=total_matched/(total_matched+total_FN)
    print("precision: %f"%(prec))
    print("recall: %f"%(rec))
    print("Matches Edges: %d (%.2f%% of GT labels) (%.2f%% of Pred labels) (%.2f%% of Matched labels)"%(total_matched_edges,(total_matched_edges/total_number_refs)*100,(total_matched_edges/total_number_preds)*100,(total_matched_edges/total_matched)*100))
    print("Matches Classes: %d (%.2f%% of GT labels) (%.2f%% of Pred labels) (%.2f%% of Matched labels)"%(total_class_match,(total_class_match/total_number_refs)*100,(total_class_match/total_number_preds)*100,(total_class_match/total_matched)*100))


    print("\nAngle\n")
    total_angle_diffs = np.asarray(total_angle_diffs)
    median_anglediffs=utils.get_median(total_angle_diffs)
    angle_th_1 = 10
    angle_th_2 = 45
    angle_th_3 = 22.5
    angle_th_range=np.array([0,5,10,15,20,30,45])
    lessthan10_all = (total_angle_diffs<math.radians(angle_th_1)).sum()
    lessthan45_all = (total_angle_diffs<math.radians(angle_th_2)).sum()
    lessthan22_all = (total_angle_diffs<math.radians(angle_th_3)).sum()
    print("Min AngleDiff: %.2f"%(math.degrees(total_angle_diffs.min())))
    print("Mean AngleDiff: %.2f"%(math.degrees(total_angle_diffs.mean())))
    print("Median AngleDiff: %.2f"%(math.degrees(median_anglediffs)))
    print("Max AngleDiff: %.2f"%(math.degrees(total_angle_diffs.max())))
    print("Number of less than 10 degree: %d (%.2f%% of GT labels) (%.2f%% of Pred labels) (%.2f%% of Matched labels)"%(lessthan10_all,(lessthan10_all/total_number_refs)*100,(lessthan10_all/total_number_preds)*100,(lessthan10_all/total_matched)*100))
    print("Number of less than 45 degree: %d (%.2f%% of GT labels) (%.2f%% of Pred labels) (%.2f%% of Matched labels)"%(lessthan45_all,(lessthan45_all/total_number_refs)*100,(lessthan45_all/total_number_preds)*100,(lessthan45_all/total_matched)*100))
    print("Number of less than 22.5 degree: %d (%.2f%% of GT labels) (%.2f%% of Pred labels) (%.2f%% of Matched labels)"%(lessthan22_all,(lessthan22_all/total_number_refs)*100,(lessthan22_all/total_number_preds)*100,(lessthan22_all/total_matched)*100))
    # print("\nAnglediff per angle\n")
    # max_angle_diff=180
    # angle_histogram=np.zeros((max_angle_diff+1))
    # for this_angle in total_angle_diffs:
    #     angle_histogram[int(math.degrees(this_angle))]+=1
    # print(angle_histogram)

    prec1 = lessthan10_all/total_matched
    prec2 = lessthan45_all/total_matched
    print("Head accuracy? (10): %f"%(prec1+(1-prec1)/2))
    print("Head accuracy? (45): %f"%(prec2+(1-prec2)/2))
    prec=lessthan10_all/(total_number_preds)
    rec=lessthan10_all/(total_number_refs)
    prec=lessthan45_all/(total_number_preds)
    rec=lessthan45_all/(total_number_refs)

    print("Angle ranges:")
    for i_ath in range(len(angle_th_range)):
        if i_ath<len(angle_th_range)-1:
            mask_ge = total_angle_diffs>=math.radians(angle_th_range[i_ath])
            mask_l = total_angle_diffs<math.radians(angle_th_range[i_ath+1])
            angles_in_range=(mask_ge*mask_l).sum()
            print("Angle range %d-%d: %d (%.2f of total)"%(angle_th_range[i_ath],angle_th_range[i_ath+1],angles_in_range,float(angles_in_range/total_matched)*100))
        else:
            mask_ge = total_angle_diffs>=math.radians(angle_th_range[i_ath])
            angles_in_range=(mask_ge).sum()
            print("Angle range %d<=: %d (%.2f of total)"%(angle_th_range[i_ath],angles_in_range,float(angles_in_range/total_matched)*100))    
        
    print("\nAngle from the common GT center\n")
    total_angle_diffs2 = np.asarray(total_angle_diffs2)
    median_anglediffs=utils.get_median(total_angle_diffs2)
    lessthan10_all = (total_angle_diffs2<math.radians(angle_th_1)).sum()
    lessthan45_all = (total_angle_diffs2<math.radians(angle_th_2)).sum()
    print("Min AngleDiff: %.2f"%(math.degrees(total_angle_diffs2.min())))
    print("Mean AngleDiff: %.2f"%(math.degrees(total_angle_diffs2.mean())))
    print("Median AngleDiff: %.2f"%(math.degrees(median_anglediffs)))
    print("Max AngleDiff: %.2f"%(math.degrees(total_angle_diffs2.max())))
    print("Number of less than 10 degree: %d (%.2f%% of GT labels) (%.2f%% of Pred labels) (%.2f%% of Matched labels)"%(lessthan10_all,(lessthan10_all/total_number_refs)*100,(lessthan10_all/total_number_preds)*100,(lessthan10_all/total_matched)*100))
    print("Number of less than 45 degree: %d (%.2f%% of GT labels) (%.2f%% of Pred labels) (%.2f%% of Matched labels)"%(lessthan45_all,(lessthan45_all/total_number_refs)*100,(lessthan45_all/total_number_preds)*100,(lessthan45_all/total_matched)*100))
    prec1 = lessthan10_all/total_matched
    prec2 = lessthan45_all/total_matched
    print("Head accuracy? (10): %f"%(prec1+(1-prec1)/2))
    print("Head accuracy? (45): %f"%(prec2+(1-prec2)/2))

    print("\nAngle-OBB\n")
    total_angle_diffs_obb = np.asarray(total_angle_diffs_obb)
    median_anglediffs=utils.get_median(total_angle_diffs_obb)
    angle_th_1 = 10
    angle_th_2 = 45
    
    lessthan10_all = (total_angle_diffs_obb<math.radians(angle_th_1)).sum()
    lessthan45_all = (total_angle_diffs_obb<math.radians(angle_th_2)).sum()
    print("Min AngleDiff: %.2f"%(math.degrees(total_angle_diffs_obb.min())))
    print("Mean AngleDiff: %.2f"%(math.degrees(total_angle_diffs_obb.mean())))
    print("Median AngleDiff: %.2f"%(math.degrees(median_anglediffs)))
    print("Max AngleDiff: %.2f"%(math.degrees(total_angle_diffs_obb.max())))
    print("Number of less than 10 degree: %d (%.2f%% of GT labels) (%.2f%% of Pred labels) (%.2f%% of Matched labels)"%(lessthan10_all,(lessthan10_all/total_number_refs)*100,(lessthan10_all/total_number_preds)*100,(lessthan10_all/total_matched)*100))
    print("Number of less than 45 degree: %d (%.2f%% of GT labels) (%.2f%% of Pred labels) (%.2f%% of Matched labels)"%(lessthan45_all,(lessthan45_all/total_number_refs)*100,(lessthan45_all/total_number_preds)*100,(lessthan45_all/total_matched)*100))
    prec1 = lessthan10_all/total_matched
    prec2 = lessthan45_all/total_matched
    print("Head accuracy? (10): %f"%(prec1+(1-prec1)/2))
    print("Head accuracy? (45): %f"%(prec2+(1-prec2)/2))
    prec=lessthan10_all/(total_number_preds)
    rec=lessthan10_all/(total_number_refs)
    prec=lessthan45_all/(total_number_preds)
    rec=lessthan45_all/(total_number_refs)

    print("\nEuclidean Center distance\n")
    total_center_dist = np.asarray(total_center_dist)
    median_center=utils.get_median(total_center_dist)
    print("Min CenterDiff: %.2f"%(total_center_dist.min()))
    print("Mean CenterDiff: %.2f"%(total_center_dist.mean()))
    print("Median CenterDiff: %.2f"%(median_center))
    print("Max CenterDiff: %.2f"%(total_center_dist.max()))

    print("\nIoU\n")
    total_ious_cat = []
    for x in total_ious: 
        total_ious_cat = [*total_ious_cat, *x]
    total_ious_cat = np.asarray(total_ious_cat)
    median_iou=utils.get_median(total_ious_cat)
    print("Min IoU: %.2f"%(total_ious_cat.min()))
    print("Mean IoU: %.2f"%(total_ious_cat.mean()))
    print("Median IoU: %.2f"%(median_iou))
    print("Max IoU: %.2f"%(total_ious_cat.max()))
