# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import DOBBModel
from ultralytics.utils import DEFAULT_CFG, RANK


class DOBBTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on an Directed Oriented Bounding Box (DOBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.obb import DOBBTrainer

        args = dict(model='yolov8n-obb.pt', data='dota8.yaml', epochs=3)
        trainer = DOBBTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a DOBBTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "dobb"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return DOBBModel initialized with specified config and weights."""
        model = DOBBModel(cfg, ch=3, nc=self.data["nc"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model
    
    def set_model_attributes(self):
        """Sets keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """Return an instance of DOBBValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "dir_loss"
        return yolo.dobb.DOBBValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
