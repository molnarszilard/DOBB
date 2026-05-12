from ultralytics import YOLO

DIRTYPE = "vector" # Direction type is a vector
DIRLOSS = "mahalanobis"
DESC = 0 # descriptor vector size: DESC*3, 

if DESC:
    model = YOLO('yolov8m-dobb_vector_descriptor.yaml')
else:
    model = YOLO('yolov8m-dobb_vector.yaml')

# Train the model
results = model.train(
    cfg='ultralytics/cfg/default.yaml',
    data='ultralytics/cfg/datasets/kitti360_pose2d_veh_build.yaml',
    # data='ultralytics/cfg/datasets/kitti360_pose2d_veh_ts.yaml',
    # data='ultralytics/cfg/datasets/7scenes_chess2di.yaml',
    mode='train',
    task='dobb',
    epochs=100, 
    imgsz=1408,
    batch=16,
    dir_gain=20,
    descgain=0,
    descriptors_size=DESC,
    class_order = 1, #10000000 for descriptors is recommended
    list_batches = True,
    cepoch=-1, # -1 no dynamic dir loss, 0 dinamic dir loss
    direction_sigma1=0.9,
    direction_sigma2=0.1,
    direction_loss=DIRLOSS,
    dir_type=DIRTYPE,
    optimizer='SGD',
    mosaic=0.0,
)