#Python code to create teh DOBB dataset from Kitti360 
# importing the required modules
import os
import cv2
import numpy as np  
import math
import argparse
import csv
import utils
import re

np.set_printoptions(suppress=True, precision=6)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--folder',default='path_to_kitti_dataset',
                    help='the root folder')
parser.add_argument('--labels',default='labels_pred/',
                    help='the labels folder')
parser.add_argument('--cameraID', default='image_00',
                    help='default camera ID')
parser.add_argument('--csv_labels_folder',default='ellipse_dir_data_gt/',
                    help='the root folder of the results')
parser.add_argument('--mode',default=2,
                    help='which classes to consider: 2-all, 0-only cars, 1-only trafficSigns')
parser.add_argument('--th', default=0.9,
                    help='threshold of the matching score, a pair should be less than this')
args = parser.parse_args()

N = 10000000
all_classes = ['building','pole','traffic light','traffic sign','person','rider','car','truck','bus','caravan','trailer','train','motorcycle','bicycle','garage','stop','smallpole','lamp','trash bin','vending machine']
chosen_classes = ['car','trafficSign']
# chosen_classes = all_classes

if args.mode==0:
    mode='car'
elif args.mode==1:
    mode='ts'
elif args.mode==2:
    mode='allclasses'

sequences = [9,10]
train_nr = [int(s) for s in re.findall(r'\d+',args.labels)][0]
images='all_for_val/images/test'
csv_labels_folder=os.path.join(args.cameraID,args.csv_labels_folder)


minimum_dotp = 0
maximum_dist = 1
all_descriptors = []
all_pred_classes = []
nr_gt_annotations = 0
nr_matched_obbs = 0
nr_no_matches = 0
nr_pred_inEval = 0
frames_selected = 0
nr_of_images = 0

for i_seq in range(len(sequences)):
    seq_number = sequences[i_seq]
    ### Reading instances masks
    csv_labels=[]
    dlist=os.listdir(os.path.join(args.folder,'2013_05_28_drive_'+f'{seq_number:04d}'+'_sync',csv_labels_folder))
    dlist.sort()
    nr_of_images+=len(dlist)
    predictions = {}
    matched_ids_all = {}
    for filename in dlist:
        if filename.endswith(".csv"):
            csv_labels.append(filename)
            gt_csv_path = os.path.join(args.folder,'2013_05_28_drive_'+f'{seq_number:04d}'+'_sync',csv_labels_folder, filename)
            if not os.path.exists(os.path.join(args.folder,images,filename[:-3]+'png')):
                if not os.path.exists(os.path.join(args.folder,images,filename[:-3]+'jpg')):
                    continue
                else:
                    img_path = os.path.join(args.folder,images,filename[:-3]+'jpg')
            else:
                img_path = os.path.join(args.folder,images,filename[:-3]+'png')
            label_path = os.path.join(args.folder,args.labels,filename[:-3]+'txt')

            frame = int([int(s) for s in re.findall(r'\d+',filename)][-1])
            print("Reading: Seq: %d, Frame: %d"%(seq_number,frame))
            ellipse_csv = []
            this_unique_ids = []
            pred_classes = []
            pred_descriptors = []
            matched_ids = []
            ### Read the GT CSV files
            with open(gt_csv_path, newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=';', quotechar='|')
                counter = -1
                for row in csvreader:
                    counter+=1
                    if counter==0:
                        continue
                    for i_row in range(len(row)):
                        row[i_row] = row[i_row].replace('\"','')
                    if row[2] not in chosen_classes:
                        continue
                    R = np.array([[row[5],row[6]],[row[7],row[8]]]).astype(np.float64)
                    w = float(row[9])*2
                    h = float(row[10])*2
                    cx = float(row[11])
                    cy = float(row[12])
                    angle_obb = math.degrees(math.atan2(R[1,0],R[0,0]))
                    dpt_pointCx = float(row[13])
                    dpt_pointCy = float(row[14])
                    i_id = int(float(row[4]))
                    s_id = int(float(row[3]))
                    class_to_learn = chosen_classes.index(row[2])
                    if args.mode<2 and args.mode!=class_to_learn:
                        continue
                    object_class = class_to_learn
                    object_class*=N
                    object_class+=seq_number*(N/100)
                    object_class+=s_id*(N/10000)
                    object_class+=i_id
                    ellipse_csv.append([class_to_learn,cx,cy,w,h,angle_obb,dpt_pointCx,dpt_pointCy])
                    this_unique_ids.append(object_class)
                    nr_gt_annotations+=1
            ellipse_csv=np.asarray(ellipse_csv)
            
            ### Read images
            img_rgb = cv2.imread(img_path)
            H,W = img_rgb.shape[:2]
            ### Read predictions
            label_array = [] 
            if os.path.isfile(label_path):            
                f_label = open(label_path, "r")
                Lines_yolo = f_label.readlines()
                f_label.close()
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
                    if args.mode<2 and args.mode!=elements[0]:
                        continue
                    rect_pred = np.array((elements[1:9])).astype(np.int32)
                    rect_pred_xywhr = cv2.minAreaRect(rect_pred.reshape(4, 2))
                    if rect_pred_xywhr[1][1]>rect_pred_xywhr[1][0]:
                        rect_pred_xywhr = ((rect_pred_xywhr[0][0],rect_pred_xywhr[0][1]),(rect_pred_xywhr[1][1],rect_pred_xywhr[1][0]),rect_pred_xywhr[2]+90) # +90 is because minRectArea stuff
                    label_array.append([elements[0],rect_pred_xywhr[0][0],rect_pred_xywhr[0][1],rect_pred_xywhr[1][0],rect_pred_xywhr[1][1],rect_pred_xywhr[2],elements[9],elements[10]])
                    pred_descriptors.append(elements[11:])
                    all_descriptors.append(elements[11:])
                    pred_classes.append(elements[0])
                label_array=np.asarray(label_array)
                pred_classes=np.asarray(pred_classes)
                pred_descriptors=np.asarray(pred_descriptors)

                ### Match them by OBB
                matched_ids_inThisImg = np.zeros(len(pred_descriptors))-1
                
                ### Find matching prediction with the reference annotations
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
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                for i in range(len(matches)):
                    matched_ids_inThisImg[matches[i,1]] = this_unique_ids[matches[i,0]]
                    nr_matched_obbs+=1
                for i_m in range(len(matched_ids_inThisImg)):
                    matched_ids.append(matched_ids_inThisImg[i_m])
                matched_ids = np.asarray(matched_ids)
                matched_ids = matched_ids.astype(np.int64)

                ### Add lists to global lists
                matched_ids_all[str(frame)] = matched_ids
                predictions[str(frame)] = {"pred_classes": pred_classes, "pred_descriptors": pred_descriptors}
        else:
            continue
    if len(csv_labels)<1:
        print("%s is empty"%(os.path.join(args.folder,'2013_05_28_drive_'+f'{seq_number:04d}'+'_sync',csv_labels_folder)))
        continue

    image_pairs = []
    image_pairs_path = os.path.join(args.folder,'global_object_pairs_'+f'{seq_number:04d}'+'.csv')
    with open(image_pairs_path, newline='') as pair_csvfile:
        pair_csvreader = csv.reader(pair_csvfile, delimiter=';', quotechar='|')
        pair_counter = -1
        for pair_row in pair_csvreader:
            pair_counter+=1
            if pair_counter==0:
                continue
            for i_pair_row in range(len(pair_row)):
                pair_row[i_pair_row] = int(float(pair_row[i_pair_row].replace('\"','')))
                image_pairs.append(pair_row[i_pair_row])
    frames_list = np.unique(image_pairs)
    frames_selected+=len(frames_list)
    pred_descriptors = []
    pred_classes = []
    matched_ids = []
    for frame in frames_list:
        if (not str(frame) in predictions) or (not str(frame) in matched_ids_all):
            continue
        for pred_d in predictions[str(frame)]["pred_descriptors"]:
            pred_descriptors.append(pred_d)
        for pred_c in predictions[str(frame)]["pred_classes"]:
            pred_classes.append(pred_c)
        for m_id in matched_ids_all[str(frame)]:
            matched_ids.append(m_id)
  
    pred_descriptors = np.asarray(pred_descriptors).reshape(-1,48)
    pred_classes = np.asarray(pred_classes).reshape(-1)
    matched_ids = np.asarray(matched_ids).reshape(-1)
    nr_pred_inEval +=len(pred_descriptors)

    pred_classes_matrix = np.repeat(np.expand_dims(pred_classes,0),len(pred_classes),axis=0)
    class_mask = pred_classes_matrix!=pred_classes_matrix.T
    matched_ids_matrix = np.repeat(np.expand_dims(matched_ids,0),len(matched_ids),axis=0)
    matched_ids_mask = matched_ids_matrix==matched_ids_matrix.T
    matched_ids_mask[np.eye(len(matched_ids_mask))>0]=False
    descriptors_products = abs(np.dot(pred_descriptors,pred_descriptors.T))

    descriptors_products[np.eye(len(pred_descriptors))>0]=minimum_dotp
    descriptors_products[class_mask]=minimum_dotp
    desc_matches = np.ones(len(descriptors_products))*(-1)

    ### Calculate top10 list, by dot product
    top10_distributions = np.zeros(10)
    descriptors_products_best = descriptors_products.copy()
    for i10 in range(10):
        best_args = np.zeros((2,len(descriptors_products_best)))
        best_args[0,:] = range(len(descriptors_products_best))
        best_args[1,:] = np.argmax(descriptors_products_best, axis=-1)
        best_args = best_args.astype(np.int32)
        top10_distributions[i10] = matched_ids_mask[best_args[0,:],best_args[1,:]].sum()
        descriptors_products_best[best_args[0,:],best_args[1,:]] = minimum_dotp

print("Total number of frames: %d"%(nr_of_images))  
print("Total number of GT objects: %d"%(nr_gt_annotations))
print("Total number of predicted objects: %d"%(len(all_descriptors)))
print("Number of matches by OBB: %d (considering by selection %d)"%(nr_matched_obbs,nr_pred_inEval))
print("Selected frames: %d"%(frames_selected))

print("Top 10 matching only by ID: ")
print(top10_distributions)
print(top10_distributions.sum())
