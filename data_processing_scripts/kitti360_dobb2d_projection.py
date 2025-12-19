#Python code to create the DOBB dataset from Kitti360. You have to run this file for each sequences.
# importing the required modules
import os
import cv2
import numpy as np  
import math
from multiprocessing import Process
import argparse
import csv
import pandas as pd
from scipy.spatial.transform import Rotation
import geometry_utils
import utils
import re
from kitti360scripts.helpers.annotation  import Annotation3D
from kitti360scripts.helpers.project import CameraPerspective as Camera
from kitti360scripts.devkits.convertOxtsPose.python.data import loadPoses
from kitti360scripts.devkits.convertOxtsPose.python.utils import postprocessPoses

np.set_printoptions(suppress=True, precision=6)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--kitti_root',default='path_to_kitti_dataset',
                    help='the root of the kitti folder of original data')
parser.add_argument('--sequence', default='2013_05_28_drive_0009_sync',
                    help='the sequence')
parser.add_argument('--cameraID', default='image_00',
                    help='default camera ID')
parser.add_argument('--multiprocess', default=1,type=int,
                    help='number of parallell processes, might result in errors')
parser.add_argument('--result_folder',default='path_to_kitti_dataset',
                    help='the root folder of the results')
parser.add_argument('--csv_labels_folder',default='ellipse_data/',
                    help='the root folder of the results')
parser.add_argument('--new_csv_labels_folder',default='ellipse_dir_data_gt/',
                    help='the root folder of the results')
parser.add_argument('--only_frame', default=-1,type=int,
                    help='only process the frame with this number (negative number turns this option off)')
parser.add_argument('--plotBB', default=False, action='store_true',
                    help='do you want to plot the 3D BB in the image?')
args = parser.parse_args()

N = 1000
chosen_classes = ['car','truck','bus','caravan','trailer','train','building']
seq_number = [int(s) for s in re.findall(r'\d+',args.sequence)][-1]

kitti_image = os.path.join(args.kitti_root,'data_2d_raw',args.sequence,args.cameraID,'data_rect')
### Creating Output structure
base=os.path.join(args.result_folder,args.sequence)
new_images='images/'
new_plots='plots'
new_labels='labels/'
csv_labels_folder=os.path.join(args.cameraID,args.csv_labels_folder)
new_csv_labels_folder=os.path.join(args.cameraID,args.new_csv_labels_folder)
if not os.path.exists(os.path.join(base,new_labels)):
    os.makedirs(os.path.join(base,new_labels))
if not os.path.exists(os.path.join(base,new_images)):
    os.makedirs(os.path.join(base,new_images))
if not os.path.exists(os.path.join(base,new_plots)):
    os.makedirs(os.path.join(base,new_plots))
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
[ts, poses] = loadPoses(os.path.join(args.kitti_root,'data_poses',args.sequence, 'poses.txt'))
poses = postprocessPoses(poses)
cam2velo = np.array([[0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],[-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],[-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],[0.0,0.0,0.0,1.0]])
cam2imu = np.array([[0.0371783278,-0.0986182135,0.9944306009,1.5752681039],[0.9992675562,-0.0053553387,-0.0378902567,0.0043914093],[0.0090621821,0.9951109327,0.0983468786,-0.6500000000]])

def process(filenames):
    for filename in filenames:
        frame = int(os.path.splitext(filename)[0])
        # if frame%10:
        #     continue
        if args.only_frame>=0 and frame!=args.only_frame:
            continue
        # if frame>2:
        #     exit()

        print("Frame: %d"%(frame))
        ### camera_tr --> Tr(cam_0 -> world)
        valid_key_found = False
        valid_key = frame
        while not valid_key_found:
            try:
                camera_tr = camera.cam2world[valid_key]
                valid_key_found=True
            except:
                valid_key-=1
        camera_R = camera_tr[:3, :3]
        camera_T = camera_tr[:3, 3]
        Rc2w_1D = np.reshape(camera_R,9)
        camera_K = camera.K
        camera_height = camera.height
        camera_width = camera.width
        # current_imu_pose = poses[np.where(ts == valid_key)[0][0]]
        
        if not os.path.exists(os.path.join(kitti_image,filename[:-3]+'png')):
            exit()
        else:
            img_rgb = cv2.imread(os.path.join(kitti_image,filename[:-3]+'png'))
        H,W = img_rgb.shape[:2]
        new_label_path = os.path.join(base,new_labels,'s'+f'{seq_number:02d}'+'_'+filename[:-3]+'txt')
        new_image_path = os.path.join(base,new_images,'s'+f'{seq_number:02d}'+'_'+filename[:-3]+'png')
        new_plot_path = os.path.join(base,new_plots,'s'+f'{seq_number:02d}'+'_'+filename[:-3]+'png')
        new_csv_path = os.path.join(base,new_csv_labels_folder,filename)
        cv2.imwrite(new_image_path,img_rgb)
        fl = open(new_label_path, "w")
        
        new_csv = []
        one_object = False
        with open(os.path.join(base,csv_labels_folder, filename), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            counter = -1
            for row in csvreader:                
                counter+=1
                if counter==0: ### Reading and appending the header
                    row.append('directionX')
                    row.append('directionY')
                    # row.append('Rc2w-11')
                    # row.append('Rc2w-12')
                    # row.append('Rc2w-13')
                    # row.append('Rc2w-21')
                    # row.append('Rc2w-22')
                    # row.append('Rc2w-23')
                    # row.append('Rc2w-31')
                    # row.append('Rc2w-32')
                    # row.append('Rc2w-33')
                    row.append('rotZ')
                    row.append('rotY')
                    row.append('rotX')

                    new_csv.append(row)
                    continue
                
                i_id = int(float(row[4]))
                s_id = int(float(row[3]))
                obj = annotation3D(s_id, i_id, frame)
                if obj:
                    if not obj.name in chosen_classes:
                        continue

                    ### Simple 3 rotation decomposition
                    rotZ,rotY,rotX = geometry_utils.decompose_camera_rotation(camera_R,order='ZYX')
                    rotZwimu = np.asarray(Rotation.from_euler('Z', rotZ, degrees=True).as_matrix())
                    rotYwimu = np.asarray(Rotation.from_euler('Y', rotY, degrees=True).as_matrix())
                    rotXwimu = np.asarray(Rotation.from_euler('X', rotX, degrees=True).as_matrix())
                    rotIMU = np.matmul(rotYwimu,rotXwimu)
                    cam2world_norm = np.matmul(rotZwimu,rotIMU)
                    # Rc2w_1D = np.reshape(cam2world_norm,9)

                    ### 4 rotation decomposition
                    # rotX90 = np.asarray(Rotation.from_euler('X', 90, degrees=True).as_matrix())
                    # cam2world2 = np.matmul(camera_R,rotX90)
                    # rotZ,rotY,rotX = geometry_utils.decompose_camera_rotation(cam2world2,order='ZYX')
                    # rotZwimu = np.asarray(Rotation.from_euler('Z', rotZ, degrees=True).as_matrix())
                    # rotYwimu = np.asarray(Rotation.from_euler('Y', rotY, degrees=True).as_matrix())
                    # rotXwimu = np.asarray(Rotation.from_euler('X', rotX, degrees=True).as_matrix())
                    # rotIMU = np.matmul(rotYwimu,rotXwimu)
                    # cam2world_norm = np.matmul(rotZwimu,rotIMU,rotX90.T)
                    # Rc2w_1D = np.reshape(cam2world_norm,9)
                    
                    ### Checking the matrix decomposition
                    # unitX=np.array([1.0,0.0,0.0])
                    # vecX1=rotIMU@unitX
                    # vecX2=camera_R@unitX
                    # ang=math.degrees(angle_between(vecX1,vecX2))
                    # print(camera_R)
                    # print(np.matmul(rotZwimu,rotIMU)@camera_R.T)
                    # print(rotZ,rotY,rotX)
                    # print(math.radians(rotZ),math.radians(rotY),math.radians(rotX))
                    # print(vecX1,vecX2)
                    # print(ang)

                    camX1w=np.matmul(camera_R,np.array([1.0,0.0,0.0])) ### Camera unit vector in world
                    phi=math.atan2(camX1w[1],camX1w[0]) ### Camera rotation in world (should be the same as rotZ)

                    objX1w=np.matmul(obj.R,np.array([1.0,0.0,0.0])) ### Object direction vector in world
                    objX1w[2]*=0 ### In 2d we need to drop the Z coordinate to the XY plane
                    objX1w = utils.unit_vector(objX1w) ### obj.R has scaling built in, let's create a unit vector
                    theta=math.atan2(objX1w[1],objX1w[0]) ### Object rotation in world around the Z coordinate
                    objX1c=np.matmul(rotZwimu.T,objX1w) ### Object direction vector in camera (same Z)
                    # print(objX1c)
                    objX1c = utils.unit_vector(objX1c)
                    angle_dpt = math.atan2(objX1c[1],objX1c[0]) #alpha, this is the direction angle, that the DOBB needs to learn

                    ### Print the ellipsoid coordinates for the MATLAB plotting
                    ### distance = math.sqrt((obj.T[0]-camera_tr[0,-1])**2+(obj.T[1]-camera_tr[1,-1])**2)
                    rot = np.asarray(Rotation.from_euler('Z', rotZ+90, degrees=True).as_matrix()) ## The coordinates needs to be in Velodyne coordinates, where Y is forward
                    new_vW=obj.T-camera_T
                    new_vC=rot.T@new_vW+[0.79,0.3,-0.18]
                    rot_w = np.asarray(Rotation.from_euler('Z', -theta, degrees=False).as_matrix())
                    vert = (rot_w@obj.vertices.T).T
                    this_ellipse = (new_vC[0],new_vC[1],new_vC[2],(vert[:,1].max()-vert[:,1].min())/2, (vert[:,0].max()-vert[:,0].min())/2, (vert[:,2].max()-vert[:,2].min())/2,math.degrees(angle_dpt),0,0) # Translation, Scale, Rotation
                    # print(this_ellipse)

                    ### PLOTTING THE AXES ###
                    # r0 = Rotation.identity()
                    # ax = plt.figure().add_subplot(projection="3d", proj_type="ortho")
                    # plot_rotated_axes(ax, r0, name="W", offset=(0, 0, 0))                    
                    # plot_rotated_axes(ax, Rotation.from_matrix(camera_R), name="c_o", offset=(2, 0, 0))
                    # plot_rotated_axes(ax, Rotation.from_matrix(np.matmul(camera_R,rotIMU)), name="", offset=(4, 0, 0))
                    # plot_rotated_axes(ax, Rotation.from_matrix(np.matmul(rotIMU,camera_R)), name="", offset=(6, 0, 0))
                    # plot_rotated_axes(ax, Rotation.from_matrix(np.matmul(camera_R,rotIMU.T)), name="", offset=(8, 0, 0))
                    # plot_rotated_axes(ax, Rotation.from_matrix(np.matmul(rotIMU.T,camera_R)), name="", offset=(10, 0, 0))
                    # # plot_rotated_axes(ax, Rotation.from_matrix(np.matmul(rotIMU,cam2world2)), name="c", offset=(6, 0, 0))
                    # ### Plotting the original direction angle into the world on each coordinate system
                    # ax.plot([0.0,0.0+objX1w[0]/np.max(abs(objX1w))], [0.0,objX1w[1]/np.max(abs(objX1w))], [0.0,objX1w[2]/np.max(abs(objX1w))], "#ff00ca")
                    # # ax.plot([2.0,2.0+objX1w[0]/np.max(abs(objX1w))], [0.0,objX1w[1]/np.max(abs(objX1w))], [0.0,objX1w[2]/np.max(abs(objX1w))], "#ff00ca")
                    # # ax.plot([4.0,4.0+objX1w[0]/np.max(abs(objX1w))], [0.0,objX1w[1]/np.max(abs(objX1w))], [0.0,objX1w[2]/np.max(abs(objX1w))], "#ff00ca")
                    # # ax.plot([6.0,6.0+objX1w[0]/np.max(abs(objX1w))], [0.0,objX1w[1]/np.max(abs(objX1w))], [0.0,objX1w[2]/np.max(abs(objX1w))], "#ff00ca")
                    # # ax.plot([8.0,8.0+objX1w[0]/np.max(abs(objX1w))], [0.0,objX1w[1]/np.max(abs(objX1w))], [0.0,objX1w[2]/np.max(abs(objX1w))], "#ff00ca")
                    # # ax.plot([0.0,0.0+objXctow[0]/np.max(abs(objXctow))], [0.0,objXctow[1]/np.max(abs(objXctow))], [0.0,objXctow[2]/np.max(abs(objXctow))], "#ff8900")
                    # # ax.plot([2.0,2.0+objXctow[0]/np.max(abs(objXctow))], [0.0,objXctow[1]/np.max(abs(objXctow))], [0.0,objXctow[2]/np.max(abs(objXctow))], "#ff8900")
                    # # ax.plot([4.0,4.0+objXctow[0]/np.max(abs(objXctow))], [0.0,objXctow[1]/np.max(abs(objXctow))], [0.0,objXctow[2]/np.max(abs(objXctow))], "#ff8900")
                    # # ax.plot([6.0,6.0+objXctow[0]/np.max(abs(objXctow))], [0.0,objXctow[1]/np.max(abs(objXctow))], [0.0,objXctow[2]/np.max(abs(objXctow))], "#ff8900")
                    # # ax.plot([0.0,objXctow[0]/np.max(abs(objXctow))], [0.0,objXctow[1]/np.max(abs(objXctow))], [0.0,objXctow[2]/np.max(abs(objXctow))], "#ffa400")
                    # loc = np.array([(6, 0, 0), (6, 0, 0)])
                    # # ax.plot([6.0,6.0], [0.0,0.0], [0.0,-1.0], "#a400ff")
                    # ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), zlim=(-1.25, 1.25))
                    # ax.set(xticks=range(-1, 12), yticks=[-1, 0, 1], zticks=[-1, 0, 1])
                    # ax.set_aspect("equal", adjustable="box")
                    # ax.figure.set_size_inches(6, 5)
                    # plt.tight_layout()
                    # plt.show()   

                    ### ADD DIRECTION ELEMENTS TO CSV ###
                    # row.append(math.degrees(angle_dpt))
                    for elem in objX1c[:2]:
                        row.append(elem)
                    # for elem in Rc2w_1D:
                    #     row.append(elem)
                    row.append(math.radians(rotZ))
                    row.append(math.radians(rotY))
                    row.append(math.radians(rotX))
                    new_csv.append(row)

                    ### CREATE THE TXT FILE FOR YOLO TRAINING ###
                    R = np.array([[row[5],row[6]],[row[7],row[8]]]).astype(np.float64)
                    w = float(row[9])*2
                    h = float(row[10])*2
                    cx = float(row[11])
                    cy = float(row[12])
                    angle_obb = math.atan2(R[1,0],R[0,0])
                    corners = utils.xywhr2xyxyxyxy(np.array([cx,cy,w,h,angle_obb]))
                    
                    if obj.name=='building':
                        object_class = 1
                    else:
                        object_class = 0
                    fl.write("%d %f %f %f %f %f %f %f %f %f %f\n"%(object_class,corners[0,0]/W,corners[0,1]/H,corners[1,0]/W,corners[1,1]/H,corners[2,0]/W,corners[2,1]/H,corners[3,0]/W,corners[3,1]/H,objX1c[0],objX1c[1]))
                    if not one_object:
                        a=min(w,h)/2
                        dx=cx+a*objX1c[0] ### Using pixels the rotation is inverse (angle_dpt is calculated around Z, while in this case it is around -Z)
                        dy=cy+a*(-objX1c[1])
                        # one_object=True ### This is in the case when we want to plot only the first object into an image
                        img_rgb = cv2.ellipse(img_rgb, (int(cx),int(cy)), (int(w/2),int(h/2)), math.degrees(angle_obb), 0, 360, utils.get_color('orange'), 2)
                        img_rgb = cv2.arrowedLine(img_rgb, (int(cx),int(cy)), (int(dx),int(dy)), utils.get_color('red'), 2)
                    # Tr_obj2world = np.array([[obj.R[0,0],obj.R[0,1],obj.R[0,1],obj.T[0]],[obj.R[1,0],obj.R[1,1],obj.R[1,1],obj.T[1]],[obj.R[2,0],obj.R[2,1],obj.R[2,1],obj.T[2]],[0.0,0.0,0.0,1.0]])

                    ### PLOTTING 3D BB in the image
                    if args.plotBB:
                        new_vertices_cam = np.zeros_like(obj.vertices)
                        new_vertices_velo = np.zeros_like(obj.vertices)
                        new_projected_points = np.zeros((len(obj.vertices),2)).astype(np.int32)
                        fx = camera_K[0,0]
                        fy = camera_K[1,1]
                        x0 = camera_K[0,2]
                        y0 = camera_K[1,2]
                        for i in range(len(obj.vertices)):
                            new_vertices_cam[i]=np.matmul(np.linalg.inv(camera_tr),np.concatenate((obj.vertices[i],np.array([1.0]))))[:3]
                            z = new_vertices_cam[i,2]
                            u = (new_vertices_cam[i,0]*fx)/z
                            v = (new_vertices_cam[i,1]*fy)/z
                            pixel_pos_x = round(u + x0)
                            pixel_pos_y = round(v + y0)
                            if pixel_pos_x < 0:                    
                                pixel_pos_x = -pixel_pos_x                        
                            if pixel_pos_x > camera_width - 1:                    
                                pixel_pos_x = camera_width - 1                        
                            if pixel_pos_y < 0:
                                pixel_pos_y = -pixel_pos_y                        
                            if pixel_pos_y > camera_height - 1:
                                pixel_pos_y = camera_height - 1                        
                            new_projected_points[i,0] = pixel_pos_x
                            new_projected_points[i,1] = pixel_pos_y                        
                            new_vertices_velo[i]=np.matmul(cam2velo,np.concatenate((new_vertices_cam[i],np.array([1.0]))))[:3] ### In case you need the coordinates in the velodyne
                            # img_rgb = cv2.circle(img_rgb, (pixel_pos_x,pixel_pos_y), 1, utils.get_color(counter), 1)
                        # for i in range(len(obj.lines)):
                        #     img_rgb = cv2.line(img_rgb, (new_projected_points[obj.lines[i][0],0],new_projected_points[obj.lines[i][0],1]), (new_projected_points[obj.lines[i][1],0],new_projected_points[obj.lines[i][1],1]),utils.get_color(counter), 1)

        ### Object points in the velodyne pcd: new_vertices_velo
        ### Object x vector in the velodyne pcd objX0velo,objX1velo,objX10velo
        ### Camera x vector in the velodyne pcd camX0velo,camX1velo,camX10velo
        df = pd.DataFrame(np.asarray(new_csv))
        df.to_csv(new_csv_path,header=False, index=False, sep=';', quotechar='|')
        fl.close()
        cv2.imwrite(new_plot_path,img_rgb)

def main():     
    if args.multiprocess==1:
        process(csv_labels)
    else:
        threads = min(args.multiprocess,len(csv_labels))
        batches = np.array_split(csv_labels,threads)
        active_threads = []
        for t in range(threads):
            p = Process(target=process, name='thread%d'%(t), args=(batches[t],))
            p.start()
      
if __name__ == "__main__": 
    main()