from ultralytics import YOLO

DIRTYPE = "vector" # Direction type is a vector

# loss of direction point (Options: mahalanobis,euclidean,vector,probiou,angle,kl,gma)
DIRLOSS = "mahalanobis2" ## for vwector

# Load a model
model = YOLO('runs/dobb/train/weights/best.pt')

# Validate the model
metrics = model.val(
    cfg='ultralytics/cfg/default.yaml',
    data='ultralytics/cfg/datasets/kitti360_pose2d_veh_build.yaml',
    # data='ultralytics/cfg/datasets/7scenes_chess2di.yaml',
    task='dobb',
    batch=4,
    imgsz=1408,
    dir_type=DIRTYPE
)