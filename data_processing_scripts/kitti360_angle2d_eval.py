#Python code to create teh DOBB dataset from Kitti360. You have to run this file for each sequences.
# importing the required modules
import os
import cv2
import numpy as np  
import math
import argparse
import csv
import pandas as pd

from kitti360scripts.helpers.annotation  import Annotation3D
from kitti360scripts.helpers.project import CameraPerspective as Camera
from scipy.spatial.transform import Rotation
import geometry_utils
import utils
import re

np.set_printoptions(suppress=True, precision=6)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--folder',default='path_to_kitti_dataset',
                    help='the root folder')
parser.add_argument('--labels',default='labels_pred/',
                    help='the labels folder')
parser.add_argument('--kitti_root',default='path_to_kitti_dataset',
                    help='the root of the kitti folder of original data')
parser.add_argument('--sequence', default='2013_05_28_drive_0010_sync',
                    help='the sequence')
parser.add_argument('--cameraID', default='image_00',
                    help='default camera ID')
parser.add_argument('--nrcl', default=2,
                    help='number of classes')
parser.add_argument('--csv_labels_folder',default='ellipse_dir_data_gt/',
                    help='the root folder of the results')
parser.add_argument('--new_csv_labels_folder',default='ellipse_dir_data_pred/',
                    help='the root folder of the results')
parser.add_argument('--only_frame', default=-1,type=int,
                    help='only process the frame with this number (negative number turns this option off)')
parser.add_argument('--plot_dirs', default=True, action='store_true',
                    help='plot the direction vectors in the images?')
parser.add_argument('--no_matching', default=False, action='store_true',
                    help='do you want to plot the 3D BB in the image?')
args = parser.parse_args()

seq_number = [int(s) for s in re.findall(r'\d+',args.sequence)][-1]
train_nr = [int(s) for s in re.findall(r'\d+',args.labels)][0]

### Creating Output structure
base=os.path.join(args.folder,args.sequence)
images='images/'
suspicious_images='sus_images/'
new_plots='plots'
csv_labels_folder=os.path.join(args.cameraID,args.csv_labels_folder)
new_csv_labels_folder=os.path.join(args.cameraID,args.new_csv_labels_folder)

if not os.path.exists(os.path.join(base,new_plots)):
    os.makedirs(os.path.join(base,new_plots))
if not os.path.exists(os.path.join(base,suspicious_images)):
    os.makedirs(os.path.join(base,suspicious_images))
if not os.path.exists(os.path.join(base,new_csv_labels_folder)):
    os.makedirs(os.path.join(base,new_csv_labels_folder))

### Reading instances masks
csv_labels=[]
dlist=os.listdir(os.path.join(base,csv_labels_folder))
dlist.sort()
for filename in dlist:
    if filename.endswith(".csv"):
        csv_labels.append(filename)
    else:
        continue
if len(csv_labels)<1:
    print("%s is empty"%(os.path.join(base,csv_labels_folder)))
    exit()

### Reading 3D bounding boxes
kitti_3dbboxes = os.path.join(args.kitti_root,'data_3d_bboxes')
annotation3D = Annotation3D(kitti_3dbboxes, args.sequence)
camera = Camera(root_dir=args.kitti_root, seq=args.sequence)

def process():
    frames_class0=open(os.path.join(base,'kitti360_dobb_'+str(train_nr)+'_s'+f'{seq_number:02d}'+'_'+'frames_class_0.txt'), "w")
    frames_class1=open(os.path.join(base,'kitti360_dobb_'+str(train_nr)+'_s'+f'{seq_number:02d}'+'_'+'frames_class_1.txt'), "w")
    frames_classall=open(os.path.join(base,'kitti360_dobb_'+str(train_nr)+'_s'+f'{seq_number:02d}'+'_'+'frames_class_all.txt'), "w")
    frames_classboth=open(os.path.join(base,'kitti360_dobb_'+str(train_nr)+'_s'+f'{seq_number:02d}'+'_'+'frames_class_both.txt'), "w")
    objects_class0 = 0
    objects_class1 = 0
    every_gt_dir_angle_cl0 = []
    every_pred_dir_angle_cl0 = []
    every_gt_dir_angle_cl1 = []
    every_pred_dir_angle_cl1 = []
    every_pred_phi = []
    total_angle_diffs_camera = []
    total_angle_diffs_rot = []
    total_angle_diffs_world = []
    total_angle_diffs_conversion = []
    min_errors = []
    row=[]
    row.append("ImageName")
    row.append("angle_error")
    min_errors.append(row)
    for filename in csv_labels:
        frame_matched_cl0=False
        frame_matched_cl1=False
        frame = int(os.path.splitext(filename)[0])
        # if frame%10:
        #     continue
        if args.only_frame>=0 and frame!=args.only_frame:
            continue
        # if frame>2:
        #     exit()

        print("Frame: %d"%(frame))
        valid_key_found = False
        valid_key = frame
        while not valid_key_found:
            try:
                camera_tr = camera.cam2world[valid_key]
                valid_key_found=True
            except:
                valid_key-=1
        camera_R = camera_tr[:3, :3]
        img_rgb = cv2.imread(os.path.join(base,images,'s'+f'{seq_number:02d}'+'_'+filename[:-3]+'png'))
        H,W = img_rgb.shape[:2]
        label_path = os.path.join(base,args.labels,'s'+f'{seq_number:02d}'+'_'+filename[:-3]+'txt')
        new_plot_path = os.path.join(base,new_plots,'s'+f'{seq_number:02d}'+'_'+filename[:-3]+'png')
        new_sus_images_path = os.path.join(base,suspicious_images,'s'+f'{seq_number:02d}'+'_'+filename[:-3]+'png')
        new_csv_path = os.path.join(base,new_csv_labels_folder,filename)
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
                        elements[i]=float(elements[i])*W
                    else:
                        elements[i]=float(elements[i])*H
                else:
                    elements[i]=float(elements[i])
            rect_pred = np.array((elements[1:9])).astype(np.int32)
            rect_pred_xywhr = cv2.minAreaRect(rect_pred.reshape(4, 2))
            if rect_pred_xywhr[1][1]>rect_pred_xywhr[1][0]:
                rect_pred_xywhr = ((rect_pred_xywhr[0][0],rect_pred_xywhr[0][1]),(rect_pred_xywhr[1][1],rect_pred_xywhr[1][0]),rect_pred_xywhr[2]+90) # +90 is because minRectArea stuff
            label_array.append([elements[0],rect_pred_xywhr[0][0],rect_pred_xywhr[0][1],rect_pred_xywhr[1][0],rect_pred_xywhr[1][1],rect_pred_xywhr[2],elements[9],elements[10]])

        base_csv = []
        ellipse_csv = []
        new_csv = []
        ids = [] ### contains the semanticID and the instanceID
        with open(os.path.join(base,csv_labels_folder, filename), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            counter = -1
            for row in csvreader:
                counter+=1
                if counter==0:
                    if not args.no_matching:   
                        row[15]="theta"
                    base_csv.append(row)
                    continue
                ids.append([int(float(row[3])),int(float(row[4]))])
                base_csv.append(row)
                R = np.array([[row[5],row[6]],[row[7],row[8]]]).astype(np.float64)
                w = float(row[9])*2
                h = float(row[10])*2
                cx = float(row[11])
                cy = float(row[12])
                angle_obb = math.degrees(math.atan2(R[1,0],R[0,0]))
                dpt_pointCx = float(row[13])
                dpt_pointCy = float(row[14])
                if row[2]=='building':
                    ellipse_csv.append([1,cx,cy,w,h,angle_obb,dpt_pointCx,dpt_pointCy])
                else:
                    ellipse_csv.append([0,cx,cy,w,h,angle_obb,dpt_pointCx,dpt_pointCy])
        base_csv=np.asarray(base_csv)
        new_csv.append(base_csv[0])
        ellipse_csv=np.asarray(ellipse_csv)

        IOUs = np.zeros((len(ellipse_csv),len(label_array)))
        for Ri in range(len(ellipse_csv)):
            for Pi in range(len(label_array)):
                if ellipse_csv[Ri][0]==label_array[Pi][0]:
                    center = (label_array[Pi][1],label_array[Pi][2])
                    size = (label_array[Pi][3],label_array[Pi][4])
                    rect_pred_xywhr = (center,size,label_array[Pi][5])
                    center = (ellipse_csv[Ri][1],ellipse_csv[Ri][2])
                    size = (ellipse_csv[Ri][3],ellipse_csv[Ri][4])
                    rect_ref_xywhr = (center,size,ellipse_csv[Ri][5])
                    r1 = cv2.rotatedRectangleIntersection(rect_ref_xywhr, rect_pred_xywhr)
                    if r1[0] != 0:
                        rect_ref_xywhr_ar = np.array((rect_ref_xywhr[0][0],rect_ref_xywhr[0][1],rect_ref_xywhr[1][0],rect_ref_xywhr[1][1],rect_ref_xywhr[2]))
                        rect_pred_xywhr = np.array((rect_pred_xywhr[0][0],rect_pred_xywhr[0][1],rect_pred_xywhr[1][0],rect_pred_xywhr[1][1],rect_pred_xywhr[2]))
                        rect_ref_xywhr_ar[4] = math.radians(rect_ref_xywhr_ar[4])
                        rect_pred_xywhr[4] = math.radians(rect_pred_xywhr[4])
                        iou = utils.probiou(rect_ref_xywhr_ar,rect_pred_xywhr)
                        if iou>=0.5:
                            IOUs[Ri,Pi]=iou
        matches = np.nonzero(IOUs >= 0.5)  # IoU > threshold and classes match
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[IOUs[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        
        ### Simple 3 rotation decomposition
        rotZ,rotY,rotX = geometry_utils.decompose_camera_rotation(camera_R,order='ZYX')
        every_pred_phi.append(rotZ)
        rotZwimu = np.asarray(Rotation.from_euler('Z', rotZ, degrees=True).as_matrix())
        rotYwimu = np.asarray(Rotation.from_euler('Y', rotY, degrees=True).as_matrix())
        rotXwimu = np.asarray(Rotation.from_euler('X', rotX, degrees=True).as_matrix())
        rotIMU = np.matmul(rotYwimu,rotXwimu)
        cam2world_norm = np.matmul(rotZwimu,rotIMU)
        
        objects_classs0_this_frame = 0
        objects_classs1_this_frame = 0
        image_rot_error = 360
        sus_img=False
        for i in range(len(matches)):
            cx,cy,w,h,angle_obb,dpt_pointCx_gt,dpt_pointCy_gt = ellipse_csv[matches[i,0]][1:]
            cxp,cyp,wp,hp,angle_obbp,dpt_pointCx_pred,dpt_pointCy_pred = label_array[matches[i,1]][1:]
            if label_array[matches[i,1]][0]==0:
                frame_matched_cl0=True
                objects_classs0_this_frame +=1   
            else:
                frame_matched_cl1=True
                objects_classs1_this_frame +=1
            
            base_csv[matches[i,0]+1][9] = wp/2
            base_csv[matches[i,0]+1][10] = hp/2
            base_csv[matches[i,0]+1][11] = cxp
            base_csv[matches[i,0]+1][12] = cyp
            base_csv[matches[i,0]+1][5] = math.cos(math.radians(angle_obbp))
            base_csv[matches[i,0]+1][6] = -math.sin(math.radians(angle_obbp))
            base_csv[matches[i,0]+1][7] = math.sin(math.radians(angle_obbp))
            base_csv[matches[i,0]+1][8] = math.cos(math.radians(angle_obbp))
            
            dpt_angle_pred=math.atan2(dpt_pointCy_pred,dpt_pointCx_pred)
            base_csv[matches[i,0]+1][13]=math.cos(dpt_angle_pred)
            base_csv[matches[i,0]+1][14]=math.sin(dpt_angle_pred)

            dpt_angle_gt=math.atan2(dpt_pointCy_gt,dpt_pointCx_gt)
            if label_array[matches[i,1]][0]==0:
                every_gt_dir_angle_cl0.append(math.degrees(dpt_angle_gt))
                every_pred_dir_angle_cl0.append(math.degrees(dpt_angle_pred))
            else:
                every_gt_dir_angle_cl1.append(math.degrees(dpt_angle_gt))
                every_pred_dir_angle_cl1.append(math.degrees(dpt_angle_pred))
            a=min(w,h)/2
            px=cx+a*dpt_pointCx_gt
            py=cy-a*dpt_pointCy_gt # Rotation changes from Z to -Z in the images

            a=min(wp,hp)/2
            pxp=cxp+a*dpt_pointCx_pred
            pyp=cyp-a*dpt_pointCy_pred

            s_id,i_id = ids[matches[i,0]]
            obj = annotation3D(s_id, i_id, frame)
            objX1w = np.matmul(obj.R,np.array([1.0,0.0,0.0]))
            objX1w = utils.unit_vector(objX1w)
            theta = math.atan2(objX1w[1],objX1w[0])

            phiZ = theta-dpt_angle_pred
            rotZphiZ = np.asarray(Rotation.from_euler('Z', phiZ, degrees=False).as_matrix())
            Rc2w_pred = np.matmul(rotZphiZ,rotYwimu,rotXwimu)
            Rc2w_pred_1D = np.reshape(Rc2w_pred,9)    
            if not args.no_matching:   
                base_csv[matches[i,0]+1][15:18]=[theta,math.radians(rotY),math.radians(rotX)]

            new_csv.append(base_csv[matches[i,0]+1])
            img_rgb = cv2.ellipse(img_rgb, (int(cx),int(cy)), (int(w/2),int(h/2)), angle_obb, 0, 360, utils.get_color('yellow'), 2)
            img_rgb = cv2.ellipse(img_rgb, (int(cxp),int(cyp)), (int(wp/2),int(hp/2)), angle_obbp, 0, 360, utils.get_color('purple'), 2)
            if args.plot_dirs:
                img_rgb = cv2.arrowedLine(img_rgb, (int(cx),int(cy)), (int(px),int(py)), utils.get_color('yellow'), 2)
                img_rgb = cv2.arrowedLine(img_rgb, (int(cxp),int(cyp)), (int(pxp),int(pyp)), utils.get_color('purple'), 2)

            rotZ_angle_dpt_pred = np.asarray(Rotation.from_euler('Z', dpt_angle_pred, degrees=False).as_matrix())
            rotZ_angle_dpt_gt = np.asarray(Rotation.from_euler('Z', dpt_angle_gt, degrees=False).as_matrix())

            unitCx=np.array([1.0,0.0,0.0])
            dirVectC_pred = np.array([dpt_pointCx_pred,dpt_pointCy_pred,0.0])
            dirVectC_gt = np.array([dpt_pointCx_gt,dpt_pointCy_gt,0.0])
            dirVectW_pred = np.matmul(Rc2w_pred,dirVectC_pred)
            dirVectW_gt = np.matmul(cam2world_norm,dirVectC_gt)

            unit_c2w_pred = np.matmul(Rc2w_pred,unitCx)
            unit_c2w_gt = np.matmul(camera_R,unitCx)

            ang_diff_camera = math.degrees(utils.angle_between(dirVectC_pred,dirVectC_gt))
            ang_diff_world = math.degrees(utils.angle_between(dirVectW_pred,dirVectW_gt))
            if image_rot_error>ang_diff_camera:
                image_rot_error=ang_diff_camera

            total_angle_diffs_rot.append(math.degrees(utils.angle_between(unit_c2w_pred,unit_c2w_gt)))
            total_angle_diffs_camera.append(ang_diff_camera)
            total_angle_diffs_world.append(ang_diff_world)
            total_angle_diffs_conversion.append(abs(ang_diff_camera-ang_diff_world))
            # base_csv[matches[i,0]+1][19]=ang_diff_camera

            if abs(ang_diff_camera)>150:
                sus_img = True

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
        if sus_img:
            cv2.imwrite(new_sus_images_path,img_rgb)
        df = pd.DataFrame(np.asarray(new_csv))
        df.to_csv(new_csv_path,header=False, index=False, sep=';', quotechar='|')
        cv2.imwrite(new_plot_path,img_rgb)
        if frame_matched_cl0:
            frames_class0.write("%d\n"%(frame))
            if args.nrcl==1:
                objects_class0+=objects_classs0_this_frame
        if frame_matched_cl1:
            frames_class1.write("%d\n"%(frame))
        if frame_matched_cl1 or frame_matched_cl0:
            frames_classall.write("%d\n"%(frame))
        if frame_matched_cl1 and frame_matched_cl0:
            frames_classboth.write("%d\n"%(frame))
            row=[]
            row.append(f'{frame:010d}.png')
            row.append(image_rot_error)
            min_errors.append(row)
            if frame_matched_cl0:
                objects_class0+=objects_classs0_this_frame
            if frame_matched_cl1:
                objects_class1+=objects_classs1_this_frame
    df = pd.DataFrame(total_angle_diffs_camera)
    df.to_csv(os.path.join(base, 'kitti360_dobb_'+str(train_nr)+'_s'+f'{seq_number:02d}'+'_'+"angle_diff_camera.csv"),header=False, index=False)

    df = pd.DataFrame(every_gt_dir_angle_cl0)
    df.to_csv(os.path.join(base, 'kitti360_dobb_'+str(train_nr)+'_s'+f'{seq_number:02d}'+'_'+"every_gt_dir_angle_cl0.csv"),header=False, index=False)

    df = pd.DataFrame(every_pred_dir_angle_cl0)
    df.to_csv(os.path.join(base, 'kitti360_dobb_'+str(train_nr)+'_s'+f'{seq_number:02d}'+'_'+"every_pred_dir_angle_cl0.csv"),header=False, index=False)

    df = pd.DataFrame(every_gt_dir_angle_cl1)
    df.to_csv(os.path.join(base, 'kitti360_dobb_'+str(train_nr)+'_s'+f'{seq_number:02d}'+'_'+"every_gt_dir_angle_cl1.csv"),header=False, index=False)

    df = pd.DataFrame(every_pred_dir_angle_cl1)
    df.to_csv(os.path.join(base, 'kitti360_dobb_'+str(train_nr)+'_s'+f'{seq_number:02d}'+'_'+"every_pred_dir_angle_cl1.csv"),header=False, index=False)

    df = pd.DataFrame(every_pred_phi)
    df.to_csv(os.path.join(base, 'kitti360_dobb_'+str(train_nr)+'_s'+f'{seq_number:02d}'+'_'+"every_pred_phi_camera_rot.csv"),header=False, index=False)

    df = pd.DataFrame(total_angle_diffs_world)
    df.to_csv(os.path.join(base, 'kitti360_dobb_'+str(train_nr)+'_s'+f'{seq_number:02d}'+'_'+"angle_diff_world.csv"),header=False, index=False)

    df = pd.DataFrame(total_angle_diffs_conversion)
    df.to_csv(os.path.join(base, 'kitti360_dobb_'+str(train_nr)+'_s'+f'{seq_number:02d}'+'_'+"angle_diff_conversion.csv"),header=False, index=False)

    df = pd.DataFrame(total_angle_diffs_rot)
    df.to_csv(os.path.join(base, 'kitti360_dobb_'+str(train_nr)+'_s'+f'{seq_number:02d}'+'_'+"angle_diff_rotations.csv"),header=False, index=False)

    df = pd.DataFrame(np.asarray(min_errors))
    df.to_csv(os.path.join(base,'kitti360_dobb_'+str(train_nr)+'_s'+f'{seq_number:02d}'+'_'+'minimum_errors.csv',),header=False, index=False, sep=';', quotechar='|')

    frames_class0.close()
    frames_class1.close()
    frames_classall.close()
    frames_classboth.close()
    print("Objects in the common images of class 0 (vehicle): %d"%(objects_class0))
    print("Objects in the common images of class 1 (building): %d"%(objects_class1))

if __name__ == "__main__": 
    process() 
