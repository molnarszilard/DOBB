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
import pandas as pd

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

minimum_dotp = -1
maximum_dist = 2
all_descriptors = []
all_pred_classes = []
correct_match_scores = []
incorrect_match_scores = []
correct_match_scores_onlybest = []
incorrect_match_scores_onlybest = []
nr_gt_annotations = 0
nr_pred_inAllPairs = 0
nr_matched_obbs = 0
nr_no_matches = 0
number_of_pairs = 0
correct_pairs = []
correct_pairs_onlybest = []

for i_seq in range(len(sequences)):
    seq_number = sequences[i_seq]
    ### Reading instances masks
    csv_labels=[]
    dlist=os.listdir(os.path.join(args.folder,'2013_05_28_drive_'+f'{seq_number:04d}'+'_sync',csv_labels_folder))
    dlist.sort()
    matched_ids_all = {}
    predictions = {}
    predicted_labels = {}
    all_images = {}
    for filename in dlist:
        if filename.endswith(".csv"):
            csv_labels.append(filename)
            gt_csv_path = os.path.join(args.folder,'2013_05_28_drive_'+f'{seq_number:04d}'+'_sync',csv_labels_folder, filename)
            frame = int([int(s) for s in re.findall(r'\d+',filename)][-1])
            print("Reading: Seq: %d, Frame: %d"%(seq_number,frame))
            ellipse_csv = []
            this_unique_ids = []
            pred_classes = []
            pred_descriptors = []
            matched_ids = []
            ### Verify that everything exists
            if not os.path.isfile(gt_csv_path):
                print("GT Label file does not exists: %s"%(gt_csv_path))
            if not os.path.exists(os.path.join(args.folder,images,filename[:-3]+'png')):
                if not os.path.exists(os.path.join(args.folder,images,filename[:-3]+'jpg')):
                    continue
                else:
                    img_path = os.path.join(args.folder,images,filename[:-3]+'jpg')
            else:
                img_path = os.path.join(args.folder,images,filename[:-3]+'png')
            if not os.path.isfile(img_path):
                print("Image file does not exists: %s"%(img_path))
            label_path = os.path.join(args.folder,args.labels,filename[:-3]+'txt')
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
                all_images[str(frame)] = {"img_rgb": img_rgb, "seq": f'{seq_number:04d}'}
                predicted_labels[str(frame)] = label_array
        else:
            continue
    if len(csv_labels)<1:
        print("%s is empty"%(os.path.join(args.folder,'2013_05_28_drive_'+f'{seq_number:04d}'+'_sync',csv_labels_folder)))
        continue

    image_pairs = []
    image_pairs_path = os.path.join(args.folder,'object_pairs_between_image_pairs_'+f'{seq_number:04d}'+'.csv')
    with open(image_pairs_path, newline='') as pair_csvfile:
        pair_csvreader = csv.reader(pair_csvfile, delimiter=';', quotechar='|')
        pair_counter = -1
        for pair_row in pair_csvreader:
            pair_counter+=1
            if pair_counter==0:
                continue
            for i_pair_row in range(len(pair_row)):
                pair_row[i_pair_row] = int(float(pair_row[i_pair_row].replace('\"','')))
            image_pairs.append(pair_row)
            number_of_pairs +=1
            pred_descriptors = []
            pred_classes = []
            matched_ids = []
            pred_labels = []
            image_pair = []
            len_first = 0
            for i_pair_row in range(1,len(pair_row)):
                frame = pair_row[i_pair_row]
                if (not str(frame) in predictions) or (not str(frame) in matched_ids_all):
                    continue
                for pred_d in predictions[str(frame)]["pred_descriptors"]:
                    pred_descriptors.append(pred_d)
                for pred_c in predictions[str(frame)]["pred_classes"]:
                    pred_classes.append(pred_c)
                for m_id in matched_ids_all[str(frame)]:
                    matched_ids.append(m_id)
                for pred_l in predicted_labels[str(frame)]:
                    pred_labels.append(pred_l)
                for im in all_images[str(frame)]:
                    image_pair.append(im)
                if i_pair_row==1:
                    len_first=len(pred_descriptors)
                
            ###
            if len(pred_descriptors)<2:
                continue
            
            pred_descriptors = np.asarray(pred_descriptors).reshape(-1,48)
            pred_classes = np.asarray(pred_classes).reshape(-1)
            matched_ids = np.asarray(matched_ids).reshape(-1)
            nr_pred_inAllPairs +=len(pred_descriptors)

            pred_classes_matrix = np.repeat(np.expand_dims(pred_classes,0),len(pred_classes),axis=0)
            class_mask = pred_classes_matrix!=pred_classes_matrix.T
            descriptors_products = abs(np.dot(pred_descriptors,pred_descriptors.T))

            descriptors_products[np.eye(len(pred_descriptors))>0]=minimum_dotp
            descriptors_products[class_mask]=minimum_dotp
            desc_matches = np.ones(len(descriptors_products))*(-1)

            descriptors_products_best = descriptors_products.copy()
            for i_d in range(len(descriptors_products_best)):
                if descriptors_products_best.max()<=0:
                    continue
                best_id = np.argmax(descriptors_products_best)
                best_x = int(best_id/len(descriptors_products_best))
                best_y = int(best_id%len(descriptors_products_best))
                desc_matches[best_x]=best_y
                desc_matches[best_y]=best_x
                descriptors_products_best[best_x,:] = minimum_dotp
                descriptors_products_best[:,best_y] = minimum_dotp
                descriptors_products_best[best_y,:] = minimum_dotp
                descriptors_products_best[:,best_x] = minimum_dotp

            desc_matches = desc_matches.astype(np.int64)
            descriptors_products_best = descriptors_products.copy()
            for i_d in range(len(desc_matches)):
                if desc_matches[i_d]==-1:
                    nr_no_matches+=1
                else:
                    best_dist = maximum_dist-descriptors_products[i_d,desc_matches[i_d]]
                    descriptors_products_best[i_d,desc_matches[i_d]] = minimum_dotp
                    descriptors_products_best[i_d,descriptors_products_best[i_d]>=descriptors_products[i_d,desc_matches[i_d]]] = minimum_dotp
                    ### What if there is only one other object?
                    if np.max(descriptors_products_best[i_d])<0:
                        match_scores = maximum_dist-descriptors_products[i_d,desc_matches[i_d]]
                        this_id = matched_ids[i_d]
                        pair_id = matched_ids[desc_matches[i_d]]
                        if this_id>=0:
                            if this_id==pair_id:
                                correct_match_scores_onlybest.append(match_scores)
                                correct_pairs_onlybest.append([seq_number,pair_row[1],pair_row[2],this_id])
                            else:
                                incorrect_match_scores_onlybest.append(match_scores)
                        else:
                            nr_no_matches+=1
                    else:
                        sec_best_desc = maximum_dist-np.max(descriptors_products_best[i_d])
                        match_scores = best_dist/sec_best_desc

                        this_id = matched_ids[i_d]
                        pair_id = matched_ids[desc_matches[i_d]]
                        this_seq = int((this_id%N)/(N/100))
                        if this_id>=0:
                            if seq_number!=this_seq:
                                print("seeq not matched")
                            if this_id==pair_id:
                                correct_match_scores.append(match_scores)
                                correct_pairs.append([seq_number,pair_row[1],pair_row[2],this_id])
                            else:
                                incorrect_match_scores.append(match_scores)
                        else:
                            nr_no_matches+=1
                                
print("Number of image pairs: %d"%(number_of_pairs))
print("Total number of GTs: %d"%(nr_gt_annotations))
print("Total number of preds: %d"%(len(all_descriptors)))
print("Number of matches by OBB (by pairs): %d (%d)"%(nr_matched_obbs,nr_pred_inAllPairs))
print("Correctly matched descriptors by dotP and OBB: %d (%.2f%%)"%(len(correct_match_scores)+len(correct_match_scores_onlybest),(len(correct_match_scores)+len(correct_match_scores_onlybest))/nr_pred_inAllPairs*100))
print("InCorrectly matched descriptors by dotP and OBB: %d (%.2f%%)"%(len(incorrect_match_scores)+len(incorrect_match_scores_onlybest),(len(incorrect_match_scores)+len(incorrect_match_scores_onlybest))/nr_pred_inAllPairs*100))
print("Not matched descriptors by dotP and OBB: %d (%.2f%%)"%(nr_no_matches,nr_no_matches/nr_pred_inAllPairs*100))

print("\n\nFor descriptors with at least 2 matches\n\n")
correct_match_scores = np.asarray(correct_match_scores)
sorted_match_scores = np.sort(correct_match_scores)
matching_score_th = sorted_match_scores[round(len(sorted_match_scores)*0.9)]
print("The matching score threshold: %f"%(matching_score_th))
match_scores_th_below = correct_match_scores[correct_match_scores<matching_score_th]
match_scores_th_above = correct_match_scores[correct_match_scores>matching_score_th]
match_scores_th_exact = correct_match_scores[correct_match_scores==matching_score_th]
print("Score is less than threshold: %d (%.2f%%)"%(len(match_scores_th_below),len(match_scores_th_below)/len(correct_match_scores)*100))
print("Score is more than threshold: %d (%.2f%%)"%(len(match_scores_th_above),len(match_scores_th_above)/len(correct_match_scores)*100))
print("Score is exactly as the threshold: %d (%.2f%%)"%(len(match_scores_th_exact),len(match_scores_th_exact)/len(correct_match_scores)*100))

print("Correct Matching scores - Min: %f"%(correct_match_scores.min()))
print("Correct Matching scores - Mean: %f"%(correct_match_scores.mean()))
print("Correct Matching scores - Median: %f"%(utils.get_median(correct_match_scores)))
print("Correct Matching scores - Max: %f"%(correct_match_scores.max()))

incorrect_match_scores = np.asarray(incorrect_match_scores)
print("InCorrect Matching scores - Min: %f"%(incorrect_match_scores.min()))
print("InCorrect Matching scores - Mean: %f"%(incorrect_match_scores.mean()))
print("InCorrect Matching scores - Median: %f"%(utils.get_median(incorrect_match_scores)))
print("InCorrect Matching scores - Max: %f"%(incorrect_match_scores.max()))

print("\n\nDescriptors with only one match\n\n")

correct_match_scores_onlybest = np.asarray(correct_match_scores_onlybest)

sorted_match_scores_onlybest = np.sort(correct_match_scores_onlybest)
matching_score_th_onlybest = sorted_match_scores_onlybest[round(len(sorted_match_scores_onlybest)*0.9)]
print("The matching score threshold: %f"%(matching_score_th_onlybest))
match_scores_th_below_onlybest = correct_match_scores_onlybest[correct_match_scores_onlybest<matching_score_th_onlybest]
match_scores_th_above_onlybest = correct_match_scores_onlybest[correct_match_scores_onlybest>matching_score_th_onlybest]
match_scores_th_exact_onlybest = correct_match_scores_onlybest[correct_match_scores_onlybest==matching_score_th_onlybest]
print("Score is less than threshold: %d (%.2f%%)"%(len(match_scores_th_below_onlybest),len(match_scores_th_below_onlybest)/len(correct_match_scores_onlybest)*100))
print("Score is more than threshold: %d (%.2f%%)"%(len(match_scores_th_above_onlybest),len(match_scores_th_above_onlybest)/len(correct_match_scores_onlybest)*100))
print("Score is exactly as the threshold: %d (%.2f%%)"%(len(match_scores_th_exact_onlybest),len(match_scores_th_exact_onlybest)/len(correct_match_scores_onlybest)*100))

print("Nr of correct matches: %d"%(len(correct_match_scores_onlybest)))
print("Correct Matching scores - Min: %f"%(correct_match_scores_onlybest.min()))
print("Correct Matching scores - Mean: %f"%(correct_match_scores_onlybest.mean()))
print("Correct Matching scores - Median: %f"%(utils.get_median(correct_match_scores_onlybest)))
print("Correct Matching scores - Max: %f"%(correct_match_scores_onlybest.max()))

incorrect_match_scores_onlybest = np.asarray(incorrect_match_scores_onlybest)
print("Nr of wrong matches: %d"%(len(incorrect_match_scores_onlybest)))
print("InCorrect Matching scores - Min: %f"%(incorrect_match_scores_onlybest.min()))
print("InCorrect Matching scores - Mean: %f"%(incorrect_match_scores_onlybest.mean()))
print("InCorrect Matching scores - Median: %f"%(utils.get_median(incorrect_match_scores_onlybest)))
print("InCorrect Matching scores - Max: %f"%(incorrect_match_scores_onlybest.max()))

df = pd.DataFrame(correct_match_scores)
df.to_csv(os.path.join(args.folder,'kitti360_dobb_'+str(train_nr)+'_'+mode+'_descriptor_correct_match_scores.csv',),header=False, index=False, sep=';', quotechar='|')

df = pd.DataFrame(incorrect_match_scores)
df.to_csv(os.path.join(args.folder,'kitti360_dobb_'+str(train_nr)+'_'+mode+'_descriptor_incorrect_match_scores.csv',),header=False, index=False, sep=';', quotechar='|')

df = pd.DataFrame(correct_match_scores_onlybest)
df.to_csv(os.path.join(args.folder,'kitti360_dobb_'+str(train_nr)+'_'+mode+'_descriptor_correct_match_scores_onlybest.csv',),header=False, index=False, sep=';', quotechar='|')

df = pd.DataFrame(incorrect_match_scores_onlybest)
df.to_csv(os.path.join(args.folder,'kitti360_dobb_'+str(train_nr)+'_'+mode+'_descriptor_incorrect_match_scores_onlybest.csv',),header=False, index=False, sep=';', quotechar='|')

all_correct_match_scores = []
for i_m in correct_match_scores:
    all_correct_match_scores.append(i_m)
for i_m in ((correct_match_scores_onlybest/matching_score_th_onlybest)*matching_score_th):
    all_correct_match_scores.append(i_m)
all_correct_match_scores = np.asarray(all_correct_match_scores)

all_sorted_match_scores_onlybest = np.sort(all_correct_match_scores)
new_matching_score_th = all_sorted_match_scores_onlybest[round(len(all_correct_match_scores)*0.9)]
print("\n\nNew matching score threshold: %f\n"%(new_matching_score_th))

all_incorrect_match_scores = []
for i_m in incorrect_match_scores:
    all_incorrect_match_scores.append(i_m)
for i_m in ((incorrect_match_scores_onlybest/matching_score_th_onlybest)*matching_score_th):
    all_incorrect_match_scores.append(i_m)
all_incorrect_match_scores = np.asarray(all_incorrect_match_scores)

print("All correct matches: %d"%(len(all_correct_match_scores)))
print("All wrong matches: %d"%(len(all_incorrect_match_scores)))
print("All correct matches, below threshold: %d (%.2f%%)"%(len(all_correct_match_scores[all_correct_match_scores<new_matching_score_th]),len(all_correct_match_scores[all_correct_match_scores<new_matching_score_th])/len(all_correct_match_scores)*100))
print("All correct matches, above threshold: %d (%.2f%%)"%(len(all_correct_match_scores[all_correct_match_scores>=new_matching_score_th]),len(all_correct_match_scores[all_correct_match_scores>=new_matching_score_th])/len(all_correct_match_scores)*100))
print("All wrong matches, below threshold: %d (%.2f%%)"%(len(all_incorrect_match_scores[all_incorrect_match_scores<new_matching_score_th]),len(all_incorrect_match_scores[all_incorrect_match_scores<new_matching_score_th])/len(all_incorrect_match_scores)*100))
print("All wrong matches, above threshold: %d (%.2f%%)"%(len(all_incorrect_match_scores[all_incorrect_match_scores>=new_matching_score_th]),len(all_incorrect_match_scores[all_incorrect_match_scores>=new_matching_score_th])/len(all_incorrect_match_scores)*100))

df = pd.DataFrame(all_correct_match_scores)
df.to_csv(os.path.join(args.folder,'kitti360_dobb_'+str(train_nr)+'_'+mode+'_descriptor_all_correct_match_scores.csv',),header=False, index=False, sep=';', quotechar='|')

df = pd.DataFrame(all_incorrect_match_scores)
df.to_csv(os.path.join(args.folder,'kitti360_dobb_'+str(train_nr)+'_'+mode+'_descriptor_all_incorrect_match_scores.csv',),header=False, index=False, sep=';', quotechar='|')

correct_pairs = np.asarray(correct_pairs)
pairs_to_save = correct_pairs[correct_match_scores<matching_score_th]
pairs_to_save = np.asarray(pairs_to_save)
df = pd.DataFrame(pairs_to_save)
df.to_csv(os.path.join(args.folder,'kitti360_dobb_'+str(train_nr)+'_'+mode+'_saved_imagepairs_objectid_mathcedscore.csv',),header=False, index=False, sep=';', quotechar='|')

correct_pairs_onlybest = np.asarray(correct_pairs_onlybest)
pairs_to_save_onlybest = correct_pairs_onlybest[correct_match_scores_onlybest<matching_score_th_onlybest]
pairs_to_save_onlybest = np.asarray(pairs_to_save_onlybest)
df = pd.DataFrame(pairs_to_save_onlybest)
df.to_csv(os.path.join(args.folder,'kitti360_dobb_'+str(train_nr)+'_'+mode+'_saved_imagepairs_objectid_mathcedscore.csv',),header=False, index=False, sep=';', quotechar='|')

all_correct_pairs = []
for i_m in correct_pairs:
    all_correct_pairs.append(i_m)
for i_m in correct_pairs_onlybest:
    all_correct_pairs.append(i_m)

all_pairs_to_save = np.asarray(all_correct_pairs)[all_correct_match_scores<new_matching_score_th]

df = pd.DataFrame(all_pairs_to_save)
df.to_csv(os.path.join(args.folder,'kitti360_dobb_'+str(train_nr)+'_'+mode+'_saved_all_imagepairs_objectid_mathcedscore.csv',),header=False, index=False, sep=';', quotechar='|')

all_pairs_to_save_unique = np.unique([tuple(row) for row in all_pairs_to_save], axis=0)
df = pd.DataFrame(all_pairs_to_save_unique)
df.to_csv(os.path.join(args.folder,'kitti360_dobb_'+str(train_nr)+'_'+mode+'_saved_all_imagepairs_unique_objectid_mathcedscore.csv',),header=False, index=False, sep=';', quotechar='|')