from ultralytics import YOLO

DIRTYPE = "vector" # Direction type is a vector
DESC = 0 # Do not calculate descriptors, dataloader will do a shuffle sampler
# DESC = 16 # Calculate descriptors, dataloader requires a batches.json file in the images folder


# Load a model
model = YOLO('runs/dobb/train379/weights/best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(
    source='/mnt/ssd2/datasets/kitti360/kitti360_pose2d_descriptors_manyclasses/all_for_val/images/test/',
    task='dobb',
    # imgsz=[1242,375],
    imgsz=1408,
    # imgsz=640,
    # imgsz=1408,
    save_txt=True,
    # iou=0.75,
    iou=0.4,
    # show_labels=True,
    # save_frames=True,
    # save=True,
    # device=1,
    dir_type=DIRTYPE,
    descriptors_size=DESC,
    class_order = 10000000,
    list_batches = True,
)
