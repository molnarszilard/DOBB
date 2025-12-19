# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class DOBBPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on an Directed Oriented Bounding Box (DOBB) model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.dobb import DOBBPredictor

        args = dict(model='yolov8n-obb.pt', source=ASSETS)
        predictor = DOBBPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes DOBBPredictor with optional model and data configuration overrides."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "dobb"

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
            rotated=True,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, 6:7]], dim=-1))
            rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
            # xywh, r, conf, cls, 
            
            if self.args.dir_type in ['vector']:
                pred_dps = pred[:,pred.shape[-1]-2:pred.shape[-1]]
            else:
                pred_dps = pred[:,-1].unsqueeze(-1)
            dobb = torch.cat([rboxes, pred_dps, pred[:, 4:6], ], dim=-1)
            
            results.append(Results(orig_img, path=img_path, names=self.model.names, dobb=dobb, dir_type=self.args.dir_type))
        return results
