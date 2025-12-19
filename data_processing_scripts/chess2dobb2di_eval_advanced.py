# This file was heavily based on 3D-Aware-Ellipses-for-Visual-Localization.
## With this file you will evaluate the DOBB inference results on the 7-Scenes Chess dataset.
## You have to run this file for each sequences.

import argparse
import os

import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation
import re
import pandas as pd
import csv
import json

from dataset_loader import Dataset_loader
from scene_loader import Scene_loader
import geometry_utils
import utils

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="data_processing_scripts/7-Scenes_Chess_scene.json", help="<Required> Input Scene file (.json)")
    parser.add_argument("--dataset_test", default="data_processing_scripts/7-Scenes_Chess_dataset_test.json", help="<Required> Input Dataset file (.json)")
    parser.add_argument("--dataset_train", default="data_processing_scripts/7-Scenes_Chess_dataset_train.json", help="<Required> Input Dataset file (.json)")
    parser.add_argument("--base_dir", default="path_to_chess_dataset", help="<Required> Output annotated file (.json)")
    parser.add_argument("--norm_labels",  action="store_true", default=False,
                        help="<Optional> normalize label coordinates in [0,1]")
    parser.add_argument('--sequence', default='seq-05', ### Training: 1,4,6     Test: 2,3,5
                    help='the sequence')
    parser.add_argument('--labels',default='labels_pred311/',
                    help='the labels folder')
    parser.add_argument('--plot_dirs', default=True, action='store_true',
                    help='plot the direction vectors in the images?')
    if len(sys.argv[1:])>0:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    dataset_dir = 'yolo_dobb2di'
    base_dir = os.path.join(args.base_dir,dataset_dir,args.sequence)
    in_csv_dir = os.path.join(base_dir,'ellipse_data_gt')
    lse_inliers_csv_path = os.path.join(args.base_dir,dataset_dir,'inlier_object_ids.csv')
    in_images_dir = os.path.join(base_dir,'images')
    in_labels_dir = os.path.join(base_dir,args.labels)
    out_plot_dir = os.path.join(base_dir,'plots')
    out_plot_sota_dir = os.path.join(base_dir,'plots_sota')
    out_csv_dir = os.path.join(base_dir,'ellipse_data_pred')
    sequence_nr = [int(s) for s in re.findall(r'\d+',args.sequence)][0]
    train_nr = [int(s) for s in re.findall(r'\d+',args.labels)][0]    

    input_dataset_test_file = args.dataset_test
    input_dataset_train_file = args.dataset_train
    input_scene_file = args.scene

    if out_plot_dir is not None:
        utils.create_if_needed(out_plot_dir)
    if out_plot_sota_dir is not None:
        utils.create_if_needed(out_plot_sota_dir)
    if out_csv_dir is not None:
        utils.create_if_needed(out_csv_dir)

    # Load scene
    scene = Scene_loader(input_scene_file)

    # Load dataset
    loader_test = Dataset_loader(input_dataset_test_file)
    loader_train = Dataset_loader(input_dataset_train_file)

    if sequence_nr==1:
        seq_multi = 0
        loader=loader_train
    elif sequence_nr==2:
        seq_multi = 0
        loader=loader_test
    elif sequence_nr==3:
        seq_multi = 1000
        loader=loader_test
    elif sequence_nr==4:
        seq_multi = 1000
        loader=loader_train
    elif sequence_nr==5:
        seq_multi = 2000
        loader=loader_test
    elif sequence_nr==6:
        seq_multi = 2000
        loader=loader_train

    dobb_used_pairs = np.zeros((1000,12,2),dtype=np.uint8)
    prev_frame = 0
    max_inliers_p_frame = 0
    with open(lse_inliers_csv_path, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            counter = -1
            counter_inliers_p_frames = 0
            for row in csvreader:
                counter+=1
                if counter==0:
                    continue
                frame_indices=[int(s) for s in re.findall(r'\d+',row[0])]
                if frame_indices[0]!=sequence_nr:
                    continue
                if frame_indices[1]!=prev_frame:
                    counter_inliers_p_frames = 0
                    prev_frame = frame_indices[1]
                dobb_used_pairs[frame_indices[1],counter_inliers_p_frames,0]=int(row[1])
                dobb_used_pairs[frame_indices[1],counter_inliers_p_frames,1]=int(row[2])
                counter_inliers_p_frames+=1
                if max_inliers_p_frame<counter_inliers_p_frames:
                    max_inliers_p_frame=counter_inliers_p_frames

    sota_frames_minimum2_detections = []
    sota_compare_errors_file = os.path.join('path_to_out_errors','chess_inria_sota_seq02-03-05_rot_errors.txt')
    sota_used_ellipses_path = os.path.join('path_to_out_errors','used_ellipses.csv')
    sota_predictions_path = os.path.join('path_to_predictions','predictions.json')
    sota_used_pairs = np.zeros((3000,3,2),dtype=np.uint8)
    with open(sota_used_ellipses_path, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            counter = -1
            for row in csvreader:
                pairs=[int(s) for s in re.findall(r'\d+',row[0])]
                id = pairs[0]
                sota_used_pairs[id][0][0] = pairs[1]
                sota_used_pairs[id][0][1] = pairs[2]
                sota_used_pairs[id][1][0] = pairs[3]
                sota_used_pairs[id][1][1] = pairs[4]
                sota_used_pairs[id][2][0] = pairs[5]
                sota_used_pairs[id][2][1] = pairs[6]
    with open(sota_predictions_path) as json_data:
        sota_predictions = json.load(json_data)
        json_data.close()
    
    f_sota = open(sota_compare_errors_file, "r")
    Lines_sota = f_sota.readlines()
    f_sota.close()
    label_sota = []
    sota_detection = 0
    for line in Lines_sota:
        current_line = line[:-1]
        elements=current_line.split(" ")
        label_sota.append(elements[0])
        if float(elements[0])>=0:
            sota_detection+=1

    ### Reading instances masks
    csv_labels=[]
    dlist=os.listdir(in_csv_dir)
    dlist.sort()
    for filename in dlist:
        if filename.endswith(".csv"):
            csv_labels.append(filename)
        else:
            continue
    if len(csv_labels)<1:
        print("%s is empty"%(in_csv_dir))
        exit()
    
    nr_of_classes = 7
    nr_of_max_instances = 0
    for obj in scene:
        obj_iid = obj["object_id"]
        if obj_iid>nr_of_max_instances:
            nr_of_max_instances=obj_iid
    object_ids=np.ones((nr_of_classes,nr_of_max_instances+1),dtype=np.int8)*(-1)
    for obj_id in range(len(scene.objects)):
        obj_sid = scene.objects[obj_id]["category_id"]
        obj_iid = scene.objects[obj_id]["object_id"]
        object_ids[obj_sid][obj_iid]=obj_id

    sota_total_number_of_preds = 0
    sota_total_number_of_dets = 0
    sota_total_number_of_preds3 = 0
    sota_total_number_of_gts = 0
    sota_total_number_of_matches = 0
    sota_total_number_of_matches3 = 0
    number_of_frames = 0
    sota_total_number_of_min0 = 0
    sota_total_number_of_min1 = 0
    sota_total_number_of_min2 = 0
    sota_total_number_of_min3 = 0

    dobb_total_number_of_preds = 0
    dobb_total_number_of_preds_inlier = 0
    dobb_total_number_of_gts = 0
    dobb_total_number_of_matches = 0
    dobb_total_number_of_matches3 = 0

    frames_class0=open(os.path.join(base_dir,'chess_%s_%s_frames_class_0.txt'%(train_nr,args.sequence)), "w")
    frames_class1=open(os.path.join(base_dir,'chess_%s_%s_frames_class_1.txt'%(train_nr,args.sequence)), "w")
    frames_class2=open(os.path.join(base_dir,'chess_%s_%s_frames_class_2.txt'%(train_nr,args.sequence)), "w")
    frames_class3=open(os.path.join(base_dir,'chess_%s_%s_frames_class_3.txt'%(train_nr,args.sequence)), "w")
    frames_class4=open(os.path.join(base_dir,'chess_%s_%s_frames_class_4.txt'%(train_nr,args.sequence)), "w")
    frames_class5=open(os.path.join(base_dir,'chess_%s_%s_frames_class_5.txt'%(train_nr,args.sequence)), "w")
    frames_class6=open(os.path.join(base_dir,'chess_%s_%s_frames_class_6.txt'%(train_nr,args.sequence)), "w")
    frames_classall=open(os.path.join(base_dir,'chess_%s_%s_frames_class_all.txt'%(train_nr,args.sequence)), "w")
    frames_class_2type=open(os.path.join(base_dir,'chess_%s_%s_frames_class_2type.txt'%(train_nr,args.sequence)), "w")
    objects_class0 = 0
    objects_class1 = 0
    objects_class2 = 0
    objects_class3 = 0
    objects_class4 = 0
    objects_class5 = 0
    objects_class6 = 0
    every_gt_dir_angle_cl0 = []
    every_gt_dir_angle_cl1 = []
    every_gt_dir_angle_cl2 = []
    every_gt_dir_angle_cl3 = []
    every_gt_dir_angle_cl4 = []
    every_gt_dir_angle_cl5 = []
    every_gt_dir_angle_cl6 = []
    every_pred_dir_angle_cl0 = []
    every_pred_dir_angle_cl1 = []
    every_pred_dir_angle_cl2 = []
    every_pred_dir_angle_cl3 = []
    every_pred_dir_angle_cl4 = []
    every_pred_dir_angle_cl5 = []
    every_pred_dir_angle_cl6 = []
    every_pred_phi = []
    total_angle_diffs_camera = []
    total_angle_diffs_rot = []
    total_angle_diffs_world = []
    total_angle_diffs_conversion = []
    min_errors = []
    min_errors_sota_comp = []
    mean_errors_sota_comp = []
    row=[]
    row.append("ImageName")
    row.append("min_angle")
    min_errors.append(row)
    row=[]
    row.append("ImageName")
    row.append("DOBB")
    row.append("3DCE")
    row.append("DOBB_avgIOU")
    row.append("DOBB_avgIOU3")
    row.append("3DCE_avgIOU")
    row.append("3DCE_avgIOU3")
    row.append("DOBB_avgIOU_inliers")
    row.append("DOBB_GT")
    row.append("DOBB_Pred")
    row.append("DOBB_Matched")
    row.append("DOBB_Matched3")
    row.append("DOBB_inliers")
    row.append("3DCE_GT")
    row.append("3DCE_Det")
    row.append("3DCE_Pred")
    row.append("3DCE_Pred3")
    row.append("3DCE_Matched")
    row.append("3DCE_Matched3")
    min_errors_sota_comp.append(row)
    mean_errors_sota_comp.append(row)
    for filename in csv_labels:
        sota_local_number_of_preds = 0
        sota_local_number_of_preds3 = 0
        sota_local_number_of_dets = 0
        sota_local_number_of_gts = 0
        sota_local_number_of_matches = 0
        sota_local_number_of_matches3 = 0

        dobb_local_number_of_preds = 0
        dobb_local_number_of_preds_inlier = 0
        dobb_local_number_of_gts = 0
        dobb_local_number_of_matches = 0
        dobb_local_number_of_matches3 = 0
        frame_matched_cl0=False
        frame_matched_cl1=False
        frame_matched_cl2=False
        frame_matched_cl3=False
        frame_matched_cl4=False
        frame_matched_cl5=False
        frame_matched_cl6=False
        frame = [int(s) for s in re.findall(r'\b\d+\b',filename)][0]
        idx=frame+seq_multi
        print("Processing: seq: %s, file: %s"%(args.sequence,filename))
        input_image_path = os.path.join(in_images_dir,args.sequence+'_'+filename[:-3]+'png')
        # if args.sequence!='seq-01':
        #     continue        
        # if frame!=268:
        #     continue
        img_rgb = cv2.imread(input_image_path)
        height,width = img_rgb.shape[:2]

        width_orig, height_orig = loader.get_image_size(idx)
        input_image_path_orig = loader.get_rgb_filename(idx)
        img_rgb_orig = cv2.imread(input_image_path_orig)
        Rt = loader.get_Rt(idx)
        # print(Rt)
        R_cam=Rt[:3,:3]
        T_cam=Rt[:3,3]
        K = loader.get_K(idx)
        P0 = K @ Rt

        sota_rot_error=float(label_sota[idx])
        sota_ellipses_all = []
        detections = sota_predictions[idx]['detections']
        sota_pred_number = len(detections)
        for det_i in range(sota_pred_number):
            sota_this_ell_number = len(detections[det_i]['ellipses'])
            for ell_i in range(sota_this_ell_number):
                ellipse_json = detections[det_i]['ellipses'][ell_i]
                catID = detections[det_i]['category_id']
                sota_ellipses_all.append([catID,ellipse_json['center'][0],ellipse_json['center'][1],ellipse_json['axes'][0],ellipse_json['axes'][1],ellipse_json['angle']])
        sota_total_number_of_preds+=len(sota_ellipses_all)
        sota_local_number_of_preds+=len(sota_ellipses_all)
        sota_local_number_of_dets+=sota_pred_number
        sota_total_number_of_dets+=sota_pred_number
        if sota_pred_number<2:
            print(sota_pred_number)
        number_of_frames+=1
        if sota_pred_number==0:
            sota_total_number_of_min0+=1
        if sota_pred_number==1:
            sota_total_number_of_min1+=1
        if sota_pred_number>=2:
            sota_total_number_of_min2+=1
            sota_frames_minimum2_detections.append('/'+args.sequence+'/'+filename[:-3]+'color.png')
        if sota_pred_number>=3:
            sota_total_number_of_min3+=1

        sota_gt_ellipses = []
        sota_gt_ids = []
        for obj in scene:
            obj_sid = obj["category_id"]
            obj_iid = obj["object_id"]
            ell0 = obj["ellipsoid"].project(P0)
            axes0, angle_obb0, center0 = ell0.decompose()
            w0, h0 = axes0
            cx0, cy0 = center0
            if utils.check_center_outlier([cx0,cy0],width_orig,height_orig):
                continue
            sota_gt_ellipses.append([obj_sid,cx0,cy0,w0,h0,angle_obb0+0])
            sota_gt_ids.append([obj_sid,obj_iid])
        sota_gt_ellipses=np.asarray(sota_gt_ellipses)
        sota_total_number_of_gts+=len(sota_gt_ellipses)
        sota_local_number_of_gts+=len(sota_gt_ellipses)

        sota_sota_ids = []
        matches_sota,IOUs_sota = utils.get_matches(sota_gt_ellipses,sota_ellipses_all)
        sota_total_number_of_matches+=len(matches_sota)
        sota_local_number_of_matches+=len(matches_sota)
        ious = []
        for i_sota  in range(IOUs_sota.shape[0]):
            for j_sota  in range(IOUs_sota.shape[1]):
                if IOUs_sota[i_sota,j_sota]>0:
                    ious.append(IOUs_sota[i_sota,j_sota])
        for i_ids in range(len(matches_sota)):
            sota_sota_ids.append(sota_gt_ids[matches_sota[i_ids,0]])
        #     ious.append(IOUs_sota[matches_sota[i_ids,0],matches_sota[i_ids,1]])
        if len(ious)!=len(sota_ellipses_all):
            for i_sota  in range(len(sota_ellipses_all)-len(ious)):
                ious.append(0)
        if len(ious)>0:
            avg_iou_3dce = np.asarray(ious).mean()
        else:
            avg_iou_3dce=0
        
        if sota_rot_error>=0:
            sota_total_number_of_preds3+=3
            sota_local_number_of_preds3+=3
            sota_used_ellipses = np.zeros((3,6))
            sota_used_pairs = np.asarray(sota_used_pairs)
            ellipse_json = detections[sota_used_pairs[idx,0,0]]['ellipses'][sota_used_pairs[idx,0,1]]
            catID = detections[sota_used_pairs[idx,0,0]]['category_id']
            sota_used_ellipses[0,:] = [catID,ellipse_json['center'][0],ellipse_json['center'][1],ellipse_json['axes'][0],ellipse_json['axes'][1],ellipse_json['angle']]

            ellipse_json = detections[sota_used_pairs[idx,1,0]]['ellipses'][sota_used_pairs[idx,1,1]]
            catID = detections[sota_used_pairs[idx,1,0]]['category_id']
            sota_used_ellipses[1,:] = [catID,ellipse_json['center'][0],ellipse_json['center'][1],ellipse_json['axes'][0],ellipse_json['axes'][1],ellipse_json['angle']]

            ellipse_json = detections[sota_used_pairs[idx,2,0]]['ellipses'][sota_used_pairs[idx,2,1]]
            catID = detections[sota_used_pairs[idx,2,0]]['category_id']
            sota_used_ellipses[2,:] = [catID,ellipse_json['center'][0],ellipse_json['center'][1],ellipse_json['axes'][0],ellipse_json['axes'][1],ellipse_json['angle']]
            new_plot_sota_path = os.path.join(out_plot_sota_dir,args.sequence+'_'+filename[:-3]+'png')        

            for i_print in range(3):
                img_rgb_orig = cv2.ellipse(img_rgb_orig, (int(sota_used_ellipses[i_print,1]),int(sota_used_ellipses[i_print,2])), (int(sota_used_ellipses[i_print,3]),int(sota_used_ellipses[i_print,4])), math.degrees(sota_used_ellipses[i_print,5]), 0, 360, utils.get_color('purple'), 2)
            for i_print in range(len(sota_gt_ellipses)):
                img_rgb_orig = cv2.ellipse(img_rgb_orig, (int(sota_gt_ellipses[i_print,1]),int(sota_gt_ellipses[i_print,2])), (int(sota_gt_ellipses[i_print,3]),int(sota_gt_ellipses[i_print,4])), math.degrees(sota_gt_ellipses[i_print,5]), 0, 360, utils.get_color('yellow'), 2)
            
            cv2.imwrite(new_plot_sota_path,img_rgb_orig)
            matches_sota3,IOUs_sota3 = utils.get_matches(sota_gt_ellipses,sota_used_ellipses,threshold=0.01)
            ious3 = []
            sota_total_number_of_matches3+=len(matches_sota3)
            sota_local_number_of_matches3+=len(matches_sota3)
            if len(matches_sota3)!=3:
                print("3DCE less than 3 good ellipse")
            sota_sota3_ids = []
            for i_sota  in range(IOUs_sota3.shape[0]):
                for j_sota  in range(IOUs_sota3.shape[1]):
                    if IOUs_sota3[i_sota,j_sota]>0:
                        ious3.append(IOUs_sota3[i_sota,j_sota])
            if len(ious3)!=len(sota_used_ellipses):
                for i_sota  in range(len(sota_used_ellipses)-len(ious3)):
                    ious3.append(0)
            for i_ids in range(len(matches_sota3)):
                sota_sota3_ids.append(sota_gt_ids[matches_sota3[i_ids,0]])
            #     ious3.append(IOUs_sota3[matches_sota3[i_ids,0],matches_sota3[i_ids,1]])
            if len(ious3)>0:
                avg_iou3_3dce = np.asarray(ious3).mean()
            else:
                avg_iou3_3dce=0
        
        rotXcorr = np.asarray(Rotation.from_euler('X', 90, degrees=True).as_matrix())        
        ### Rdiff for projection
        R_cam_new=R_cam.T@rotXcorr
        phiZ,phiY,phiX = geometry_utils.decompose_camera_rotation(R_cam_new,order='ZYX')
        rotZwimu = np.asarray(Rotation.from_euler('Z', phiZ, degrees=True).as_matrix())
        rotYwimu = np.asarray(Rotation.from_euler('Y', phiY, degrees=True).as_matrix())
        rotXwimu = np.asarray(Rotation.from_euler('X', phiX, degrees=True).as_matrix())
        rotIMU = rotYwimu@rotXwimu
        unitCx=np.array([1.0,0.0,0.0])
        unit_c2w_1=rotZwimu@rotIMU.T@rotZwimu.T@R_cam_new@unitCx
        unit_c2w_2=R_cam.T@unitCx
        Z_camera_angle=math.degrees(utils.angle_between(unit_c2w_1,unit_c2w_2))
        if unit_c2w_2[2]<0:
            Z_camera_angle*=-1
        rotZcorr2 = np.asarray(Rotation.from_euler('Z', Z_camera_angle, degrees=True).as_matrix())
        R_cam_new=R_cam.T@rotZcorr2@rotXcorr
        phiZ,phiY,phiX = geometry_utils.decompose_camera_rotation(R_cam_new,order='ZYX')
        rotZwimu = np.asarray(Rotation.from_euler('Z', phiZ, degrees=True).as_matrix())
        rotYwimu = np.asarray(Rotation.from_euler('Y', phiY, degrees=True).as_matrix())
        rotXwimu = np.asarray(Rotation.from_euler('X', phiX, degrees=True).as_matrix())
        rotIMU = rotYwimu@rotXwimu
        every_pred_phi.append(phiZ)
        # cam2world_norm = rotZwimu@rotYwimu@rotXwimu@rotXcorr.T
        # H=K@rotIMU@np.linalg.inv(K)
        # H = H / H[2][2]
        # H_1D = np.reshape(H,9)
        # P1=H@K@Rt

        label_path = os.path.join(in_labels_dir,args.sequence+'_'+filename[:-3]+'txt')
        new_plot_path = os.path.join(out_plot_dir,args.sequence+'_'+filename[:-3]+'png')
        new_csv_path = os.path.join(out_csv_dir,filename)
        f_label = open(label_path, "r")
        Lines_yolo = f_label.readlines()
        f_label.close()
        label_array = []
        for line in Lines_yolo:
            current_line = line[:-1]
            elements=current_line.split(" ")
            for i in range(len(elements)):
                if i==0:
                    elements[i]=int(elements[i])
                elif i<9:
                    if i%2:
                        elements[i]=float(elements[i])*width
                    else:
                        elements[i]=float(elements[i])*height
                else:
                    elements[i]=float(elements[i])
            rect_pred = np.array((elements[1:9])).astype(np.int32)
            rect_pred_xywhr = cv2.minAreaRect(rect_pred.reshape(4, 2))
            if rect_pred_xywhr[1][1]>rect_pred_xywhr[1][0]:
                rect_pred_xywhr = ((rect_pred_xywhr[0][0],rect_pred_xywhr[0][1]),(rect_pred_xywhr[1][1],rect_pred_xywhr[1][0]),rect_pred_xywhr[2]+90) # +90 because changing w,h
            label_array.append([elements[0],rect_pred_xywhr[0][0],rect_pred_xywhr[0][1],rect_pred_xywhr[1][0],rect_pred_xywhr[1][1],rect_pred_xywhr[2],elements[9],elements[10]])
        dobb_total_number_of_preds+=len(label_array)
        dobb_local_number_of_preds+=len(label_array)
        base_csv = []
        ellipse_csv = []
        new_csv = []
        ids = [] ### contains the semanticID and the instanceID
        with open(os.path.join(in_csv_dir, filename), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            counter = -1
            for row in csvreader:
                counter+=1
                if counter==0:
                    row[22]="theta"
                    base_csv.append(row)
                    continue
                i_id = int(float(row[2]))
                s_id = int(float(row[1]))
                obj = scene.objects[object_ids[s_id][i_id]]
                ell0 = obj["ellipsoid"].project(P0)
                axes0, angle_obb0, center0 = ell0.decompose()
                w0, h0 = axes0
                cx0, cy0 = center0
                # if utils.check_center_outlier([cx0,cy0],width,height):
                #     continue
                base_csv.append(row)
                ids.append([int(float(row[1])),int(float(row[2]))])
                
                R = np.array([[row[3],row[4]],[row[5],row[6]]]).astype(np.float64)
                w = float(row[7])
                h = float(row[8])                
                cx = float(row[9])
                cy = float(row[10])
                angle_obb = math.atan2(R[1,0],R[0,0])

                H = np.array([[row[11],row[12],row[13]],[row[14],row[15],row[16]],[row[17],row[18],row[19]]]).astype(np.float64)
                if min(h,w)<2:
                    continue
                if max(h,w)>max(width,height)/2:
                    continue

                dpt_pointCx = float(row[20])
                dpt_pointCy = float(row[21])
                phiZ = float(row[22])
                phiY = float(row[23])
                phiX = float(row[24])
                Z_camera_angle=float(row[25])
                ellipse_csv.append([i_id,cx,cy,w*2,h*2,math.degrees(angle_obb),dpt_pointCx,dpt_pointCy])
        dobb_total_number_of_gts+=len(ellipse_csv)
        dobb_local_number_of_gts+=len(ellipse_csv)
        base_csv=np.asarray(base_csv)
        new_csv.append(base_csv[0])
        ellipse_csv=np.asarray(ellipse_csv)

        objects_classs0_this_frame = 0
        objects_classs1_this_frame = 0
        objects_classs2_this_frame = 0
        objects_classs3_this_frame = 0
        objects_classs4_this_frame = 0
        objects_classs5_this_frame = 0
        objects_classs6_this_frame = 0
        image_rot_error = 360
        image_rot_error_this_frame = []

        matches,IOU_DOBBi = utils.get_matches(ellipse_csv,label_array,threshold=0.001)
        matches_dobb,IOUs_dobb = utils.get_matches(ellipse_csv,label_array)
        dobb_total_number_of_matches+=len(matches)
        dobb_local_number_of_matches+=len(matches)
        ious_dobb=[]
        ious_dobb3=[]
        ious_dobb_inliers=[]
        dobb_used_pairs_list = []

        for i_sota  in range(IOU_DOBBi.shape[0]):
            for j_sota  in range(IOU_DOBBi.shape[1]):
                if IOU_DOBBi[i_sota,j_sota]>0:
                    ious_dobb.append(IOU_DOBBi[i_sota,j_sota])
        if len(ious_dobb)!=len(label_array):
            for i_sota  in range(len(label_array)-len(ious_dobb)):
                ious_dobb.append(0)

        for i_inlier in range(12):
            dobb_used_pairs_list.append((dobb_used_pairs[frame,i_inlier,0],dobb_used_pairs[frame,i_inlier,1]))
        for i in range(len(matches)):
            s_id,i_id = ids[matches[i,0]]
            
            if (s_id,i_id) in dobb_used_pairs_list:
                if sota_rot_error>=0:
                    if [s_id,i_id] in sota_sota3_ids:
                        ious_dobb3.append(IOUs_dobb[matches[i,0],matches[i,1]])
                        dobb_total_number_of_matches3+=1
                        dobb_local_number_of_matches3+=1
                ious_dobb_inliers.append(IOUs_dobb[matches[i,0],matches[i,1]])
                dobb_total_number_of_preds_inlier+=1
                dobb_local_number_of_preds_inlier+=1
            cx,cy,w,h,angle_obb,dpt_pointCx_gt,dpt_pointCy_gt = ellipse_csv[matches[i,0]][1:]
            cxp,cyp,wp,hp,angle_obbp,dpt_pointCx_pred,dpt_pointCy_pred = label_array[matches[i,1]][1:]

            base_csv[matches[i,0]+1][7] = wp/2
            base_csv[matches[i,0]+1][8] = hp/2
            base_csv[matches[i,0]+1][9] = cxp
            base_csv[matches[i,0]+1][10] = cyp
            base_csv[matches[i,0]+1][3] = math.cos(math.radians(angle_obbp))
            base_csv[matches[i,0]+1][4] = -math.sin(math.radians(angle_obbp))
            base_csv[matches[i,0]+1][5] = math.sin(math.radians(angle_obbp))
            base_csv[matches[i,0]+1][6] = math.cos(math.radians(angle_obbp))
            
            dpt_angle_pred=math.atan2(dpt_pointCy_pred,dpt_pointCx_pred)
            base_csv[matches[i,0]+1][20]=dpt_pointCx_pred
            base_csv[matches[i,0]+1][21]=dpt_pointCy_pred

            dpt_angle_gt=math.atan2(dpt_pointCy_gt,dpt_pointCx_gt)
            if s_id==0:
                every_gt_dir_angle_cl0.append(math.degrees(dpt_angle_gt))
                every_pred_dir_angle_cl0.append(math.degrees(dpt_angle_pred))
                frame_matched_cl0=True
                objects_classs0_this_frame +=1
            if s_id==1:
                every_gt_dir_angle_cl1.append(math.degrees(dpt_angle_gt))
                every_pred_dir_angle_cl1.append(math.degrees(dpt_angle_pred))
                frame_matched_cl1=True
                objects_classs1_this_frame +=1
            if s_id==2:
                every_gt_dir_angle_cl2.append(math.degrees(dpt_angle_gt))
                every_pred_dir_angle_cl2.append(math.degrees(dpt_angle_pred))
                frame_matched_cl2=True
                objects_classs2_this_frame +=1 
            if s_id==3:
                every_gt_dir_angle_cl3.append(math.degrees(dpt_angle_gt))
                every_pred_dir_angle_cl3.append(math.degrees(dpt_angle_pred))
                frame_matched_cl3=True
                objects_classs3_this_frame +=1 
            if s_id==4:
                every_gt_dir_angle_cl4.append(math.degrees(dpt_angle_gt))
                every_pred_dir_angle_cl4.append(math.degrees(dpt_angle_pred))
                frame_matched_cl4=True
                objects_classs4_this_frame +=1 
            if s_id==5:
                every_gt_dir_angle_cl5.append(math.degrees(dpt_angle_gt))
                every_pred_dir_angle_cl5.append(math.degrees(dpt_angle_pred))
                frame_matched_cl5=True
                objects_classs5_this_frame +=1 
            if s_id==6:
                every_gt_dir_angle_cl6.append(math.degrees(dpt_angle_gt))
                every_pred_dir_angle_cl6.append(math.degrees(dpt_angle_pred))
                frame_matched_cl6=True
                objects_classs6_this_frame +=1
            a=min(w,h)/2
            px=cx+a*dpt_pointCx_gt
            py=cy-a*dpt_pointCy_gt # Rotation changes from Z to -Z in the images

            a=min(wp,hp)/2

            
            obj = scene.objects[object_ids[s_id][i_id]]
            R_ell = np.asarray(obj["ellipsoid"].to_dict()['R'])
            if i_id==0: ### Front of first TV
                obj_dirW = R_ell@np.array([1,0,0]).T
                obj_dirW[2]=0
            elif i_id==1: ### Front of second TV
                obj_dirW = R_ell@np.array([0,0,1]).T
                obj_dirW[2]=0
            elif i_id==2: ### Front of first Xbox
                obj_dirW = R_ell@np.array([1,0,0]).T
                obj_dirW[2]=0
            elif i_id==3: ### Front of second Xbox
                obj_dirW = R_ell@np.array([-1,0,0]).T
                obj_dirW[2]=0
            elif i_id==4: ### Right side of first chair
                obj_dirW = R_ell@np.array([1,0,0]).T
                obj_dirW[2]=0
            elif i_id==5: ### Right side of second chair
                obj_dirW = R_ell@np.array([1,0,0]).T
                obj_dirW[2]=0
            elif i_id==6: ### Right side of third chair
                obj_dirW = R_ell@np.array([0,0,-1]).T
                obj_dirW[2]=0
            elif i_id==7: ### Right side of timer
                obj_dirW = R_ell@np.array([1,0,0]).T
                obj_dirW[2]=0
            elif i_id==8: ### Bottom of games
                obj_dirW = R_ell@np.array([1,0,0]).T
                obj_dirW[2]=0
            elif i_id==9: ### Front of Interruptor
                obj_dirW = R_ell@np.array([1,0,0]).T
                obj_dirW[2]=0
            elif i_id==10: ### Right side of Gamepad
                obj_dirW = R_ell@np.array([1,0,0]).T
                obj_dirW[2]=0
            obj_dirW=utils.unit_vector(obj_dirW)
            theta = math.atan2(obj_dirW[1],obj_dirW[0])

            phiZ_new = theta-dpt_angle_pred
            rotZphiZ = np.asarray(Rotation.from_euler('Z', phiZ_new, degrees=False).as_matrix())
            Rc2w_pred = rotZphiZ@rotYwimu@rotXwimu@rotXcorr.T
            Rc2w_pred_1D = np.reshape(Rc2w_pred,9)    
            base_csv[matches[i,0]+1][22]=theta
            newC = utils.compose((wp, hp),angle_obbp,(cxp, cyp))

            new_csv.append(base_csv[matches[i,0]+1])
            img_rgb = cv2.ellipse(img_rgb, (int(cx),int(cy)), (int(w/2),int(h/2)), angle_obb, 0, 360, utils.get_color('yellow'), 2)
            img_rgb = cv2.ellipse(img_rgb, (int(cxp),int(cyp)), (int(wp/2),int(hp/2)), angle_obbp, 0, 360, utils.get_color('purple'), 2)
            if args.plot_dirs:
                a=min(w,h)/2
                px=cx+a*dpt_pointCx_gt
                py=cy-a*dpt_pointCy_gt
                a=min(wp,hp)/2
                pxp=cxp+a*dpt_pointCx_pred
                pyp=cyp-a*dpt_pointCy_pred
                img_rgb = cv2.arrowedLine(img_rgb, (int(cx),int(cy)), (int(px),int(py)), utils.get_color('yellow'), 2)
                img_rgb = cv2.arrowedLine(img_rgb, (int(cxp),int(cyp)), (int(pxp),int(pyp)), utils.get_color('purple'), 2)

            rotZ_angle_dpt_pred = np.asarray(Rotation.from_euler('Z', dpt_angle_pred, degrees=False).as_matrix())
            rotZ_angle_dpt_gt = np.asarray(Rotation.from_euler('Z', dpt_angle_gt, degrees=False).as_matrix())

            unitCx=np.array([1.0,0.0,0.0])
            # dirVectC_pred = np.matmul(rotZ_angle_dpt_pred,unitCx)
            # dirVectC_gt = np.matmul(rotZ_angle_dpt_gt,unitCx)
            dirVectC_pred = np.array([dpt_pointCx_pred,dpt_pointCy_pred,0.0])
            dirVectC_gt = np.array([dpt_pointCx_gt,dpt_pointCy_gt,0.0])
            dirVectW_pred = np.matmul(rotZphiZ,dirVectC_pred)
            dirVectW_gt = np.matmul(rotZwimu,dirVectC_gt)

            unit_c2w_pred = np.matmul(Rc2w_pred,unitCx)
            unit_c2w_gt = np.matmul(R_cam.T,unitCx)

            ang_diff_camera = math.degrees(utils.angle_between(dirVectC_pred,dirVectC_gt))
            ang_diff_world = math.degrees(utils.angle_between(dirVectW_pred,dirVectW_gt))
            rot_error = math.degrees(utils.angle_between(unit_c2w_pred,unit_c2w_gt))
            image_rot_error_this_frame.append(ang_diff_camera)
            if image_rot_error>ang_diff_camera:
                image_rot_error=ang_diff_camera

            total_angle_diffs_rot.append(rot_error)
            total_angle_diffs_camera.append(ang_diff_camera)
            total_angle_diffs_world.append(ang_diff_world)
            total_angle_diffs_conversion.append(abs(ang_diff_camera-ang_diff_world))
            # base_csv[matches[i,0]+1][19]=ang_diff_camera

        if len(matches)!=len(ellipse_csv):
            for i in range(len(ellipse_csv)):
                if not i in matches[:,0]:
                    cx,cy,w,h,angle_obb,dpt_pointCx_gt,dpt_pointCy_gt = ellipse_csv[i][1:]
                    a=min(w,h)/2
                    px=cx+a*dpt_pointCx_gt
                    py=cy-a*dpt_pointCy_gt
                    img_rgb = cv2.ellipse(img_rgb, (int(cx),int(cy)), (int(w/2),int(h/2)), angle_obb, 0, 360, utils.get_color('red'), 2)
                    if args.plot_dirs:
                        img_rgb = cv2.arrowedLine(img_rgb, (int(cx),int(cy)), (int(px),int(py)), utils.get_color('red'), 2)
        
        if len(matches)!=len(label_array):
            for i in range(1,len(label_array)):
                if not i in matches[:,1]:
                    cxp,cyp,wp,hp,angle_obbp,dpt_pointCx_pred,dpt_pointCy_pred = label_array[i][1:]  
                    a=min(wp,hp)/2
                    pxp=cxp+a*dpt_pointCx_pred
                    pyp=cyp-a*dpt_pointCy_pred
                    img_rgb = cv2.ellipse(img_rgb, (int(cxp),int(cyp)), (int(wp/2),int(hp/2)), angle_obbp, 0, 360, utils.get_color('orange'), 2)
                    if args.plot_dirs:
                        img_rgb = cv2.arrowedLine(img_rgb, (int(cxp),int(cyp)), (int(pxp),int(pyp)), utils.get_color('orange'), 2)

        if len(ious_dobb)>0:
            avg_iou_dobb = np.asarray(ious_dobb).mean()
        else:
            avg_iou_dobb=0
        if len(ious_dobb3)>0:
            avg_iou_dobb3 = np.asarray(ious_dobb3).mean()
        else:
            avg_iou_dobb3=0
        if len(ious_dobb_inliers)>0:
            avg_iou_dobb_inlier = np.asarray(ious_dobb_inliers).mean()
        else:
            avg_iou_dobb_inlier=0
        df = pd.DataFrame(np.asarray(new_csv))
        df.to_csv(new_csv_path,header=False, index=False, sep=';', quotechar='|')
        cv2.imwrite(new_plot_path,img_rgb)
        if frame_matched_cl0:
            frames_class0.write("%d\n"%(frame))
        if frame_matched_cl1:
            frames_class1.write("%d\n"%(frame))
        if frame_matched_cl2:
            frames_class2.write("%d\n"%(frame))
        if frame_matched_cl3:
            frames_class3.write("%d\n"%(frame))
        if frame_matched_cl4:
            frames_class4.write("%d\n"%(frame))
        if frame_matched_cl5:
            frames_class5.write("%d\n"%(frame))
        if frame_matched_cl6:
            frames_class6.write("%d\n"%(frame))
        if (frame_matched_cl0+frame_matched_cl1+frame_matched_cl2+frame_matched_cl3+frame_matched_cl4+frame_matched_cl5+frame_matched_cl6)>0:
            frames_classall.write("%d\n"%(frame))
            row=[]
            row.append(args.sequence+'_'+filename[:-3]+'png')
            row.append(image_rot_error)
            min_errors.append(row)
            if sota_rot_error>=0 and avg_iou_dobb_inlier>0:
                row=[]
                row.append(args.sequence+'_'+filename[:-3]+'png')
                row.append(image_rot_error)
                row.append(sota_rot_error)
                row.append(avg_iou_dobb)
                row.append(avg_iou_dobb3)
                row.append(avg_iou_3dce)
                row.append(avg_iou3_3dce)
                row.append(avg_iou_dobb_inlier)
                row.append(dobb_local_number_of_gts)
                row.append(dobb_local_number_of_preds)
                row.append(dobb_local_number_of_matches)
                row.append(dobb_local_number_of_matches3)
                row.append(dobb_local_number_of_preds_inlier)
                row.append(sota_local_number_of_gts)
                row.append(sota_local_number_of_dets)
                row.append(sota_local_number_of_preds)
                row.append(sota_local_number_of_preds3)
                row.append(sota_local_number_of_matches)
                row.append(sota_local_number_of_matches3)
                min_errors_sota_comp.append(row)
                row[1]=np.asarray(image_rot_error_this_frame).mean()
                mean_errors_sota_comp.append(row)
            if frame_matched_cl0:
                objects_class0+=objects_classs0_this_frame
            if frame_matched_cl1:
                objects_class1+=objects_classs1_this_frame
            if frame_matched_cl2:
                objects_class2+=objects_classs2_this_frame
            if frame_matched_cl3:
                objects_class3+=objects_classs3_this_frame
            if frame_matched_cl4:
                objects_class4+=objects_classs4_this_frame
            if frame_matched_cl5:
                objects_class5+=objects_classs5_this_frame
            if frame_matched_cl6:
                objects_class6+=objects_classs6_this_frame
        if (frame_matched_cl0+frame_matched_cl1+frame_matched_cl2+frame_matched_cl3+frame_matched_cl4+frame_matched_cl5+frame_matched_cl6)>1:
            frames_class_2type.write("%d\n"%(frame))            

    df = pd.DataFrame(total_angle_diffs_camera)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_angle_diff_camera.csv"%(train_nr,args.sequence)),header=False, index=False)
    df = pd.DataFrame(every_gt_dir_angle_cl0)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_every_gt_dir_angle_cl0.csv"%(train_nr,args.sequence)),header=False, index=False)
    df = pd.DataFrame(every_gt_dir_angle_cl1)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_every_gt_dir_angle_cl1.csv"%(train_nr,args.sequence)),header=False, index=False)
    df = pd.DataFrame(every_gt_dir_angle_cl2)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_every_gt_dir_angle_cl2.csv"%(train_nr,args.sequence)),header=False, index=False)
    df = pd.DataFrame(every_gt_dir_angle_cl3)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_every_gt_dir_angle_cl3.csv"%(train_nr,args.sequence)),header=False, index=False)
    df = pd.DataFrame(every_gt_dir_angle_cl4)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_every_gt_dir_angle_cl4.csv"%(train_nr,args.sequence)),header=False, index=False)
    df = pd.DataFrame(every_gt_dir_angle_cl5)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_every_gt_dir_angle_cl5.csv"%(train_nr,args.sequence)),header=False, index=False)
    df = pd.DataFrame(every_gt_dir_angle_cl6)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_every_gt_dir_angle_cl6.csv"%(train_nr,args.sequence)),header=False, index=False)

    df = pd.DataFrame(every_pred_dir_angle_cl0)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_every_pred_dir_angle_cl0.csv"%(train_nr,args.sequence)),header=False, index=False)
    df = pd.DataFrame(every_pred_dir_angle_cl1)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_every_pred_dir_angle_cl1.csv"%(train_nr,args.sequence)),header=False, index=False)
    df = pd.DataFrame(every_pred_dir_angle_cl2)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_every_pred_dir_angle_cl2.csv"%(train_nr,args.sequence)),header=False, index=False)
    df = pd.DataFrame(every_pred_dir_angle_cl3)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_every_pred_dir_angle_cl3.csv"%(train_nr,args.sequence)),header=False, index=False)
    df = pd.DataFrame(every_pred_dir_angle_cl4)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_every_pred_dir_angle_cl4.csv"%(train_nr,args.sequence)),header=False, index=False)
    df = pd.DataFrame(every_pred_dir_angle_cl5)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_every_pred_dir_angle_cl5.csv"%(train_nr,args.sequence)),header=False, index=False)
    df = pd.DataFrame(every_pred_dir_angle_cl6)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_every_pred_dir_angle_cl6.csv"%(train_nr,args.sequence)),header=False, index=False)

    df = pd.DataFrame(every_pred_phi)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_every_pred_phi_camera_rot.csv"%(train_nr,args.sequence)),header=False, index=False)

    df = pd.DataFrame(total_angle_diffs_world)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_angle_diff_world.csv"%(train_nr,args.sequence)),header=False, index=False)

    df = pd.DataFrame(total_angle_diffs_conversion)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_angle_diff_conversion.csv"%(train_nr,args.sequence)),header=False, index=False)

    df = pd.DataFrame(total_angle_diffs_rot)
    df.to_csv(os.path.join(base_dir, "chess_%s_%s_angle_diff_rotations.csv"%(train_nr,args.sequence)),header=False, index=False)

    df = pd.DataFrame(np.asarray(min_errors))
    df.to_csv(os.path.join(base_dir,'chess_%s_%s_minimum_errors.csv'%(train_nr,args.sequence)),header=False, index=False, sep=';', quotechar='|')

    df = pd.DataFrame(np.asarray(min_errors_sota_comp))
    df.to_csv(os.path.join(base_dir,'chess_%s_%s_minimum_errors_sota_comp.csv'%(train_nr,args.sequence)),header=False, index=False, sep=';', quotechar='|')

    df = pd.DataFrame(np.asarray(mean_errors_sota_comp))
    df.to_csv(os.path.join(base_dir,'chess_%s_%s_mean_errors_sota_comp.csv'%(train_nr,args.sequence)),header=False, index=False, sep=';', quotechar='|')

    df = pd.DataFrame(np.asarray(sota_frames_minimum2_detections))
    df.to_csv(os.path.join(base_dir,'chess_%s_%s_sota_min2_detections.csv'%(train_nr,args.sequence)),header=False, index=False, sep=';', quotechar='|')

    frames_class0.close()
    frames_class1.close()
    frames_class2.close()
    frames_class3.close()
    frames_class4.close()
    frames_class5.close()
    frames_class6.close()
    frames_classall.close()
    frames_class_2type.close()
    print("Detected objects of class 0 (tv): %d"%(objects_class0))
    print("Detected objects of class 1 (xbox): %d"%(objects_class1))
    print("Detected objects of class 2 (chair): %d"%(objects_class2))
    print("Detected objects of class 3 (timer): %d"%(objects_class3))
    print("Detected objects of class 4 (games): %d"%(objects_class4))
    print("Detected objects of class 5 (interruptor): %d"%(objects_class5))
    print("Detected objects of class 6 (gamepad): %d"%(objects_class6))
    print("The SOTA method found the pose for %d frames."%(sota_detection))
    print("SOTA GT total: %d"%(sota_total_number_of_gts))
    print("SOTA 0 objects: %d"%(sota_total_number_of_min0))
    print("SOTA 1 objects: %d"%(sota_total_number_of_min1))
    print("SOTA min2 objects: %d"%(sota_total_number_of_min2))
    print("SOTA min3 objects: %d"%(sota_total_number_of_min3))
    print("SOTA pred total: %d"%(sota_total_number_of_preds))
    print("SOTA Det total: %d"%(sota_total_number_of_dets))
    print("SOTA pred total: %d"%(sota_total_number_of_preds))
    print("SOTA pred used for pose: %d"%(sota_total_number_of_preds3))
    print("SOTA matched: %d (%.2f%% of GT, %.2f%% of preds)"%(sota_total_number_of_matches,sota_total_number_of_matches*100/sota_total_number_of_gts,sota_total_number_of_matches*100/sota_total_number_of_preds))
    print("SOTA matched3: %d (%.2f%% of GT, %.2f%% of preds)"%(sota_total_number_of_matches3,sota_total_number_of_matches3*100/sota_total_number_of_gts,sota_total_number_of_matches3*100/sota_total_number_of_preds3))

    print("DOBB GT total: %d"%(dobb_total_number_of_gts))
    print("DOBB pred total: %d"%(dobb_total_number_of_preds))
    print("DOBB pred inliers: %d"%(dobb_total_number_of_preds_inlier))
    print("DOBB matched: %d (%.2f%% of GT, %.2f%% of preds)"%(dobb_total_number_of_matches,dobb_total_number_of_matches*100/dobb_total_number_of_gts,dobb_total_number_of_matches*100/dobb_total_number_of_preds))
    print("DOBB matched3: %d (%.2f%% of GT, %.2f%% of preds)"%(dobb_total_number_of_matches3,dobb_total_number_of_matches3*100/dobb_total_number_of_gts,dobb_total_number_of_matches3*100/sota_total_number_of_preds3))

    print(number_of_frames)
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
