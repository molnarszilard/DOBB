from ultralytics import YOLO

DIRTYPE = "vector" # Direction type is a vector

# loss of direction point (Options: mahalanobis,euclidean,vector,probiou,angle,kl,gma)
DIRLOSS = "mahalanobis2" ## for vwector

model = YOLO('yolov8m-dobb_vector.yaml')

# Train the model
results = model.train(
    cfg='ultralytics/cfg/default.yaml',
    # data='ultralytics/cfg/datasets/kitti360_pose2d_veh_build.yaml',
    data='ultralytics/cfg/datasets/7scenes_chess2di.yaml',
    mode='train',
    task='dobb',
    epochs=100, 
    imgsz=1408,
    batch=4,
    dir_gain=20,
    cepoch=-1, # -1 no dynamic dir loss, 0 dinamic dir loss
    direction_sigma1=0.9,
    direction_sigma2=0.1,
    direction_loss=DIRLOSS,
    dir_type=DIRTYPE,
    optimizer='SGD',
)