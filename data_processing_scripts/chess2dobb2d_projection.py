# This file was heavily based on 3D-Aware-Ellipses-for-Visual-Localization.
## With this file you will create the dataset for the 7-Scenes Chess dataset, for processing it with the DOBB method.
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
    parser.add_argument('--sequence', default='seq-06',
                    help='the sequence')
    parser.add_argument("--apply_filters",  action="store_true", default=True,
                        help="<Optional> apply filters for extreme homographies")
    parser.add_argument("--offset",  action="store_true", default=True,
                        help="<Optional> use offset when saving the homography")
    if len(sys.argv[1:])>0:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()

    base_dir = args.base_dir
    dataset_dir = 'yolo_dobb2di'
    in_csv_dir = os.path.join(base_dir,dataset_dir,args.sequence,'ellipse_data')
    out_images_dir = os.path.join(base_dir,dataset_dir,args.sequence,'images')
    out_labels_dir = os.path.join(base_dir,dataset_dir,args.sequence,'labels')
    out_plot_dir = os.path.join(base_dir,dataset_dir,args.sequence,'plots')
    out_csv_dir = os.path.join(base_dir,dataset_dir,args.sequence,'ellipse_data_gt')
    sequence_nr = [int(s) for s in re.findall(r'\b\d+\b',args.sequence)][0]

    input_dataset_test_file = args.dataset_test
    input_dataset_train_file = args.dataset_train
    input_scene_file = args.scene

    if out_images_dir is not None:
        utils.create_if_needed(out_images_dir)

    if out_labels_dir is not None:
        utils.create_if_needed(out_labels_dir)

    if out_plot_dir is not None:
        utils.create_if_needed(out_plot_dir)

    if out_csv_dir is not None:
        utils.create_if_needed(out_csv_dir)

    # Load scene
    scene = Scene_loader(input_scene_file)

    # Load dataset
    loader_test = Dataset_loader(input_dataset_test_file)
    loader_train = Dataset_loader(input_dataset_train_file)

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

    for filename in csv_labels:
        homography_extreme = False
        frame = [int(s) for s in re.findall(r'\b\d+\b',filename)][0]
        if sequence_nr==1:
            idx=frame
            loader=loader_train
        elif sequence_nr==2:
            idx=frame
            loader=loader_test
        elif sequence_nr==3:
            idx=1000+frame
            loader=loader_test
        elif sequence_nr==4:
            idx=1000+frame
            loader=loader_train
        elif sequence_nr==5:
            idx=2000+frame
            loader=loader_test
        elif sequence_nr==6:
            idx=2000+frame
            loader=loader_train
        
        width, height = loader.get_image_size(idx)
        input_image_path = loader.get_rgb_filename(idx)
        print("Processing: seq: %s, file: %s"%(args.sequence,filename))

        img_base = cv2.imread(input_image_path)
        output_text_path = os.path.join(out_labels_dir,args.sequence+'_'+filename[:-3]+'txt')
        f = open(output_text_path, "w")

        Rt = loader.get_Rt(idx)
        # print(Rt)
        R_cam=Rt[:3,:3]
        T_cam=Rt[:3,3]
        K = loader.get_K(idx)
        P0 = K @ Rt

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

        H1=K@rotIMU@rotZcorr2.T@np.linalg.inv(K)
        H1 = H1 / H1[2][2]
        H_1D = np.reshape(H1,9)
        top,bottom, left, right = utils.calc_tblr(img_base,H1,norm=False)
        R_offset = np.array([[1,0,-left],[0,1,-top],[0,0,1]])
        Ho = R_offset@H1
        Ho = Ho / Ho[2][2]
        H = Ho if args.offset else H1
        P1=H@K@Rt
        top,bottom, left, right = utils.calc_tblr(img_base,H,norm=False)
        new_height = round(bottom-top)
        new_width = round(right-left)
        tl2,tr2,bl2,br2=utils.calc_transformed_corners(img_base,H)
        new_csv = []
        size_limit = 4000
        if max(new_height,new_width)>=size_limit and not homography_extreme:
            homography_extreme=True
            print("Size limit excceded: %dx%d"%(warp_dst1.shape[:2]))
        if not homography_extreme or not args.apply_filters:
            warp_dst1 = utils.Apply_Matrix_To_Image(Ho,img_base,new_width,new_height)
            if max(abs(left),abs(top))>=size_limit/2 and not homography_extreme:
                homography_extreme=True
                print("Too much offset: %f, %f, %f, %f"%(top,bottom,left,right))
            if max(new_height,new_width)<=10000:
                warp_dst4 = utils.rev_Apply_Matrix_To_Image(Ho,warp_dst1,width,height)
                diff = np.sqrt((img_base-warp_dst4)**2).mean()                               
                if diff>2 and not homography_extreme:
                    homography_extreme=True
                    print("Wrong reverse H, diff: %f"%(diff))
            mask = cv2.cvtColor(warp_dst1, cv2.COLOR_BGR2GRAY)
            black_ratio_limit = 0.3
            black_ratio = (cv2.countNonZero(mask))/(mask.shape[0]*mask.shape[1])
            if black_ratio<black_ratio_limit and not homography_extreme:
                homography_extreme=True
                print("Wrong black_ratio: %f"%(black_ratio))
            top_edge = np.sqrt((tl2[0]-tr2[0])**2+(tl2[1]-tr2[1])**2)
            bottom_edge = np.sqrt((bl2[0]-br2[0])**2+(bl2[1]-br2[1])**2)
            left_edge = np.sqrt((tl2[0]-bl2[0])**2+(tl2[1]-bl2[1])**2)
            right_edge = np.sqrt((tr2[0]-br2[0])**2+(tr2[1]-br2[1])**2) 
            height_ratio = max(left_edge,right_edge)/min(left_edge,right_edge)
            width_ratio = max(top_edge,bottom_edge)/min(top_edge,bottom_edge)
            wh_ratio_limit = 1/0.3
            if max(height_ratio,width_ratio)>wh_ratio_limit and not homography_extreme:
                homography_extreme=True
                print("Wrong wh ratio: %f, %f"%(width_ratio,height_ratio))
            minEdge=min(top_edge,bottom_edge,left_edge,right_edge)
            maxEdge=max(top_edge,bottom_edge,left_edge,right_edge)
            orig_edge_ratio = 640/480
            edge_ratio_limit = 5
            edge_ratio = maxEdge/minEdge
            if edge_ratio>edge_ratio_limit and not homography_extreme:
                homography_extreme=True
                print("Wrong edge ratio: %f"%(edge_ratio))

        if homography_extreme and args.apply_filters:
            f.close()
            f = open(output_text_path, "w")
            f.close()
            output_image_path1 = os.path.join(out_images_dir,args.sequence+'_'+filename[:-3]+'png')
            cv2.imwrite(output_image_path1,np.zeros_like(img_base))
            output_plot_path1 = os.path.join(out_plot_dir,args.sequence+'_'+filename[:-3]+'png')
            cv2.imwrite(output_plot_path1,np.zeros_like(img_base))
            with open(os.path.join(in_csv_dir, filename), newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=';', quotechar='|')
                counter = -1
                for row in csvreader: 
                    counter+=1
                    if row[-1]=='':
                        row=row[:-1]
                    if counter==0: ### Reading and appending the header
                        for ir in range(len(row)):
                            row[ir]=row[ir].replace(" ", "")
                        row.append('directionX')
                        row.append('directionY')
                        # row.append('directionZ')
                        row.append('rotZ')
                        row.append('rotY')
                        row.append('rotX')
                        row.append('rotZcorrection')
                        new_csv.append(row)
            df = pd.DataFrame(np.asarray(new_csv))
            new_csv_path = os.path.join(out_csv_dir,filename)
            df.to_csv(new_csv_path,header=False, index=False, sep=';', quotechar='|')
            continue
        warp_dst1_plot = warp_dst1.copy()
        new_height,new_width = warp_dst1.shape[:2]

        
        with open(os.path.join(in_csv_dir, filename), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            counter = -1
            for row in csvreader: 
                if homography_extreme:
                    continue
                counter+=1
                if row[-1]=='':
                    row=row[:-1]
                if counter==0: ### Reading and appending the header
                    for ir in range(len(row)):
                        row[ir]=row[ir].replace(" ", "")
                    row.append('directionX')
                    row.append('directionY')
                    # row.append('directionZ')
                    row.append('rotZ')
                    row.append('rotY')
                    row.append('rotX')
                    row.append('rotZcorrection')
                    new_csv.append(row)
                    continue
                i_id = int(float(row[2]))
                s_id = int(float(row[1]))                

                ### CREATE THE TXT FILE FOR YOLO TRAINING ###
                R = np.array([[row[3],row[4]],[row[5],row[6]]]).astype(np.float64)
                w = float(row[7])
                h = float(row[8])
                cx = float(row[9])
                cy = float(row[10])
                angle_obb = math.atan2(R[1,0],R[0,0])
                if h>w:
                    h,w=w,h
                    angle_obb+=math.pi/2

                H = np.array([[row[11],row[12],row[13]],[row[14],row[15],row[16]],[row[17],row[18],row[19]]]).astype(np.float64)
                
                obj = scene.objects[object_ids[s_id][i_id]]
                ell0 = obj["ellipsoid"].project(P0)
                axes0, angle_obb0, center0 = ell0.decompose()
                w0, h0 = axes0
                cx0, cy0 = center0

                ell0n = obj["ellipsoid"].project(P1)
                axes0n, angle_obb0n, center0n = ell0n.decompose()
                w0n, h0n = axes0n
                cx0n, cy0n = center0n

                angle_diff = abs(math.atan2(math.sin(angle_obb - angle_obb0n),math.cos(angle_obb - angle_obb0n))) 
                if abs(math.degrees(angle_diff))>10 and abs(math.degrees(angle_diff))<170:
                    angle_obb+=math.pi/2
                if utils.check_center_outlier([cx0,cy0],width,height):
                    continue                
                if min(h,w)<2:
                    continue
                if max(h,w)>max(warp_dst1.shape[1],warp_dst1.shape[0])/2:
                    continue

                if not args.offset:
                    cx-=left
                    cy-=top
                if utils.check_center_outlier([cx,cy],new_width,new_height):
                    continue
                corners = utils.xywhr2xyxyxyxy(np.array([cx-1,cy-1,w*2,h*2,angle_obb]))

                row[3] = math.cos(angle_obb)
                row[4] = -math.sin(angle_obb)
                row[5] = math.sin(angle_obb)
                row[6] = math.cos(angle_obb)
                
                warp_dst1_plot = cv2.ellipse(warp_dst1_plot, (int(cx),int(cy)), (int(w),int(h)), math.degrees(angle_obb), 0, 360, utils.get_color(s_id), 2)

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
                obj_dirC = rotZwimu.T@obj_dirW
                a=min(h,w)
                dx=cx+a*obj_dirC[0]
                dy=cy-a*obj_dirC[1]
                warp_dst1_plot = cv2.arrowedLine(warp_dst1_plot, (int(cx),int(cy)), (int(dx),int(dy)), utils.get_color('red'), 2)
                label_elements = [corners[0,0]/new_width,corners[0,1]/new_height,corners[1,0]/new_width,corners[1,1]/new_height,corners[2,0]/new_width,corners[2,1]/new_height,corners[3,0]/new_width,corners[3,1]/new_height,obj_dirC[0],obj_dirC[1]]
                
                if args.norm_labels:
                    label_elements=utils.norm_labels(label_elements,new_width,new_height)
                f.write("%d "%(i_id))
                for el_i in range(len(label_elements)-1):
                    f.write("%f "%(label_elements[el_i]))
                f.write("%f\n"%(label_elements[-1]))
                row.append(obj_dirC[0])
                row.append(obj_dirC[1])
                row.append(math.radians(phiZ))
                row.append(math.radians(phiY))
                row.append(math.radians(phiX-90))
                row.append(math.radians(-Z_camera_angle))
                new_csv.append(row)
        if not homography_extreme:
            f.close()

            output_image_path1 = os.path.join(out_images_dir,args.sequence+'_'+filename[:-3]+'png')
            cv2.imwrite(output_image_path1,warp_dst1)
            output_plot_path1 = os.path.join(out_plot_dir,args.sequence+'_'+filename[:-3]+'png')
            cv2.imwrite(output_plot_path1,warp_dst1_plot)
            
            df = pd.DataFrame(np.asarray(new_csv))
            new_csv_path = os.path.join(out_csv_dir,filename)
            df.to_csv(new_csv_path,header=False, index=False, sep=';', quotechar='|')

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
