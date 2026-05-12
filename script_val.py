from ultralytics import YOLO

DIRTYPE = "vector" # Direction type is a vector

DIRLOSS = "mahalanobis"

DESC = 16 # descriptor vector size: DESC*3, 

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
    dir_type=DIRTYPE,
    descriptors_size=DESC,
    class_order = 10000000,
    list_batches = True,
)