import os
import argparse
import cv2
import numpy as np
import math
import utils
from shapely import geometry

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--labels',required=True, default=None,
                    help='the directory to the labels')
parser.add_argument('--images', default=None,
                    help='the directory to the images')
parser.add_argument('--plots', default=None,
                    help='the directory where to save plots')
parser.add_argument('--ellipse', default=False, action='store_true',
                    help='rectangle or ellipse?')
parser.add_argument('--arrow', default=False, action='store_true',
                    help='draw arrow?')
parser.add_argument('--thickness', default=2, type=int,
                    help='line thickness?')
parser.add_argument('--dir_type', default=1, type=int,
                    help='dir type: 0 - dpt, 1 - vector')
args = parser.parse_args()

labels=[]
dlist=os.listdir(args.labels)
dlist.sort()
for filename in dlist:
    if filename.endswith(".txt") and not filename.startswith("classes"):
        #print(os.path.join(directory, filename))
        labels.append(filename)
    else:
        continue
if len(labels)<1:
    print("%s is empty"%(args.labels))
    exit()

if args.plots is not None and not os.path.exists(args.plots):
    os.makedirs(args.plots)

if __name__ == '__main__':
    for i in range(len(labels)):
        label_name=labels[i]
        print("Processing: "+label_name)
        f_label = open(os.path.join(args.labels, label_name), "r")
        Lines = f_label.readlines()
        f_label.close()
        img = None
        # print(args.images)
        if args.images is not None:
            img_extensions=["png","jpg","tif"]
            img = None
            for img_ext in img_extensions:
                if os.path.isfile(os.path.join(args.images, label_name[:-3]+img_ext)):
                    img_name=label_name[:-3]+img_ext
                    # print(img_name)
                    img = cv2.imread(os.path.join(args.images, img_name))
            if img is None:
                print("Image %s does not exists"%(label_name[:-3]))
                continue
        for line in Lines:
            if line.startswith('#') or line.startswith('YOLO'):
                continue
            # class,x_center,y_center,bb_x,bb_y,angle
            current_line = line[:-1]
            elements=current_line.split(" ")
            # if len(elements)!=11:
            #     print(elements)
            if img is not None:
                H,W,_ = img.shape
                if len(elements)>=10:
                    if args.dir_type==0:
                        px = int(float(elements[9])*W)
                        py = int(float(elements[10])*H)
                    elif args.dir_type==1:
                        elements_float=elements.copy()
                        for i in range(len(elements_float)):
                            if i==0:
                                elements_float[i]=int(elements_float[i])
                            elif i<9:
                                if i%2:
                                    elements_float[i]=int(float(elements_float[i])*W)
                                else:
                                    elements_float[i]=int(float(elements_float[i])*H)
                        poly = geometry.Polygon(((elements_float[1],elements_float[2]),(elements_float[3],elements_float[4]),(elements_float[5],elements_float[6]),(elements_float[7],elements_float[8])))
                        if poly.area<10:
                            # print(elements)
                            # print(elements_float)
                            # print(np.asarray(elements_float[1:9]).sum())
                            # print(poly.area)
                            continue
                        rect = np.array(elements_float[1:9])
                        rect_xywhr = cv2.minAreaRect(rect.reshape(4, 2))
                        cosD=float(elements_float[9])
                        sinD=float(elements_float[10])
                        angleD=math.atan2(sinD,cosD)
                        px,py=utils.dptFromAngle(rect_xywhr[0][0],rect_xywhr[0][1],rect_xywhr[1][0],rect_xywhr[1][1],math.radians(rect_xywhr[2]),angleD)
                    elements[9]=float(px)/W
                    elements[10]=float(py)/H
                if args.ellipse:
                    if args.arrow:
                        img = utils.draw_ellipse(img,[elements],colors=['cls','yolored'],thickness=args.thickness,plot_kp="arrow",norm=True)
                    else:
                        img = utils.draw_ellipse(img,[elements],colors=['cls','yolored'],thickness=args.thickness,plot_kp="kp",norm=True)
                else:
                    if args.arrow:
                        img = utils.draw_rectangle(img,[elements],colors=['cls','cls','cls','cls','yolored'],thickness=args.thickness,plot_kp="arrow",norm=True)
                    else:
                        img = utils.draw_rectangle(img,[elements],colors=['cls','cls','cls','cls','yolored'],thickness=args.thickness,plot_kp="kp",norm=True)
        if img is not None:
            cv2.imwrite(os.path.join(args.plots,img_name),img)
            
            