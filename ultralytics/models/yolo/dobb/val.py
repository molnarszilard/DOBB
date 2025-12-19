from pathlib import Path

import torch
import cv2

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import DOBBMetrics, batch_probiou
from ultralytics.utils.plotting import output_to_rotated_target_dpt, plot_images


class DOBBValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on an Directed Oriented Bounding Box (DOBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.dobb import DOBBValidator

        args = dict(model='yolov8n-obb.pt', data='dota8.yaml')
        validator = DOBBValidator(args=args)
        validator(model=args['model'])
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize DOBBValidator and set task to 'dobb', metrics to DOBBMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "dobb"
        self.metrics = DOBBMetrics(save_dir=self.save_dir, plot=True, on_plot=self.on_plot)

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%24s" + "%11s" * 8) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)", "mAP50ohd10", "mAP50ohd45")

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes", "keypoints"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = (
                [
                    torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i],batch["keypoints"][batch["batch_idx"] == i]], dim=-1)
                    for i in range(nb)
                ]
                if self.args.save_hybrid
                else []
            )  # for autolabelling

        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")  # validation path
        self.is_dota = isinstance(val, str) and "DOTA" in val  # is COCO
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[],tp_a10=[],tp_a45=[])

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
            rotated=True,
        )

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            pbatch = self._prepare_batch(si, batch)
            cls, bbox, kpts = pbatch.pop("cls"), pbatch.pop("bbox"), pbatch.pop("keypoints")
            nl = len(cls)
            stat = dict(
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                target_cls=torch.zeros(0, device=self.device),
                tp_a10=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_a45=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            
            stat["target_cls"] = cls
            stat["target_bangle"] = bbox[:,4]

            if npr == 0:
                if nl:
                    for k in self.stats:
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]
            stat["pred_bangle"] = predn[:, 6]

            pred_d = torch.atan2(predn[:, 8],predn[:, 7])

            target_d = torch.atan2(kpts[:,0,1],kpts[:,0,0])

            # Evaluate
            if nl:
                
                stat["tp"], stat["tp_gt"], matches, probiou = self._process_batch(predn, bbox, cls)
                if stat["tp"][:,0].sum()!=stat["tp_gt"][:,0].sum():
                    print("neq")        
                
                if len(target_d.shape)==1:
                    dir_angle_diff=abs(torch.atan2(torch.sin(target_d[:, None] - pred_d),torch.cos(target_d[:, None] - pred_d)))
                else:
                    dir_angle_diff=abs(torch.atan2(torch.sin(target_d - pred_d),torch.cos(target_d - pred_d)))
                stat["tp_a10"], _, _ = self._process_batch_angle(predn, bbox, dir_angle_diff, cls,probiou,a_th=10)
                stat["tp_a45"], _, _ = self._process_batch_angle(predn, bbox, dir_angle_diff, cls,probiou,a_th=45)

                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                file = self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, pbatch["ori_shape"], file)

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 9] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class, angle, Dx, Dy.
            gt_bboxes (torch.Tensor): Tensor of shape [M, 5] representing rotated boxes.
                Each box is of the format: x1, y1, x2, y2, angle.
            gt_cls (torch.Tensor): Tensor of shape [M] representing labels.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, 6:7]], dim=-1))
        probiou=iou.clone()
        tp_pred,matches = self.match_predictions(detections[:, 5], gt_cls, iou,get_matches05=True)
        tp_gt = torch.zeros([gt_bboxes.shape[0],10],dtype=bool,device=tp_pred.device)
        tp_gt[matches[:,0]]=True
        for i_m_gt in range(len(matches[:,0])):
            while (matches[:,0]==i_m_gt).sum()==0:
                for i_gt in range(len(matches)):
                    if matches[i_gt,0]>i_m_gt:
                        matches[i_gt,0]-=1
        for i_m_pred in range(len(matches[:,1])):
            while (matches[:,1]==i_m_pred).sum()==0:
                for i_pred in range(len(matches)):
                    if matches[i_pred,1]>i_m_pred:
                        matches[i_pred,1]-=1
        return tp_pred, tp_gt, matches, probiou

    def _process_batch_angle(self, detections, gt_bboxes, dir_angle_diff, gt_cls,iou,a_th=10):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 9] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class, angle, Dx, Dy.
            gt_bboxes (torch.Tensor): Tensor of shape [M, 5] representing rotated boxes.
                Each box is of the format: x1, y1, x2, y2, angle.
            gt_cls (torch.Tensor): Tensor of shape [M] representing labels.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        tp_pred, matches = self.match_predictions_angle(detections[:, 5], gt_cls, iou,dir_angle_diff,a_th=a_th,get_matches05=True)
        tp_gt = torch.zeros([gt_bboxes.shape[0],10],dtype=bool,device=tp_pred.device)
        tp_gt[matches[:,0]]=True
        for i_m_gt in range(len(matches[:,0])):
            while (matches[:,0]==i_m_gt).sum()==0:
                for i_gt in range(len(matches)):
                    if matches[i_gt,0]>i_m_gt:
                        matches[i_gt,0]-=1
        for i_m_pred in range(len(matches[:,1])):
            while (matches[:,1]==i_m_pred).sum()==0:
                for i_pred in range(len(matches)):
                    if matches[i_pred,1]>i_m_pred:
                        matches[i_pred,1]-=1

        return tp_pred, tp_gt, matches

    def _prepare_batch(self, si, batch):
        """Prepares and returns a batch for DOBB validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        keypoints = batch["keypoints"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad, xywh=True)  # native-space labels
            if self.args.dir_type in ["dpt"]:
                keypoints[..., :2].mul_(torch.tensor(imgsz, device=self.device)[[1, 0]])
                ops.scale_coords(imgsz, keypoints[...,:2], ori_shape, ratio_pad=ratio_pad)
        return dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad, keypoints=keypoints)

    def _prepare_pred(self, pred, pbatch):
        """Prepares and returns a batch for DOBB validation with scaled and padded bounding boxes."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"], xywh=True
        )  # native-space pred
        if self.args.dir_type in ["dpt"]:
            ops.scale_coords(pbatch["imgsz"], predn[:, 7:9], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        return predn

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_rotated_target_dpt(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        rbox = torch.cat([predn[:, :4], predn[:, 7:8]], dim=-1)
        poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)
        for i, (r, b) in enumerate(zip(rbox.tolist(), poly.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(predn[i, 5].item())],
                    "score": round(predn[i, 4].item(), 5),
                    "rbox": [round(x, 3) for x in r],
                    "poly": [round(x, 3) for x in b],
                }
            )

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        gn = torch.tensor(shape)[[1, 0]]  # normalization gain whwh
        for *xywh, conf, cls, angle, Dx, Dy in predn.tolist():
            xywha = torch.tensor([*xywh, angle]).view(1, 5)
            xyxyxyxy = (ops.xywhr2xyxyxyxy(xywha) / gn).view(-1).tolist()  # normalized xywh
            Dx /= gn[0]
            Dy /= gn[1]
            line = (cls, *xyxyxyxy, Dx,Dy, conf) if save_conf else (cls, *xyxyxyxy,Dx,Dy)  # label format
            with open(file, "a") as f:
                f.write(("%g " * len(line)).rstrip() % line + "\n")

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and self.is_dota and len(self.jdict):
            import json
            import re
            from collections import defaultdict

            pred_json = self.save_dir / "predictions.json"  # predictions
            pred_txt = self.save_dir / "predictions_txt"  # predictions
            pred_txt.mkdir(parents=True, exist_ok=True)
            data = json.load(open(pred_json))
            # Save split results
            LOGGER.info(f"Saving predictions with DOTA format to {pred_txt}...")
            for d in data:
                image_id = d["image_id"]
                score = d["score"]
                classname = self.names[d["category_id"]].replace(" ", "-")
                p = d["poly"]

                with open(f'{pred_txt / f"Task1_{classname}"}.txt', "a") as f:
                    f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")
            # Save merged results, this could result slightly lower map than using official merging script,
            # because of the probiou calculation.
            pred_merged_txt = self.save_dir / "predictions_merged_txt"  # predictions
            pred_merged_txt.mkdir(parents=True, exist_ok=True)
            merged_results = defaultdict(list)
            LOGGER.info(f"Saving merged predictions with DOTA format to {pred_merged_txt}...")
            for d in data:
                image_id = d["image_id"].split("__")[0]
                pattern = re.compile(r"\d+___\d+")
                x, y = (int(c) for c in re.findall(pattern, d["image_id"])[0].split("___"))
                bbox, score, cls = d["rbox"], d["score"], d["category_id"]
                bbox[0] += x
                bbox[1] += y
                bbox.extend([score, cls])
                merged_results[image_id].append(bbox)
            for image_id, bbox in merged_results.items():
                bbox = torch.tensor(bbox)
                max_wh = torch.max(bbox[:, :2]).item() * 2
                c = bbox[:, 6:7] * max_wh  # classes
                scores = bbox[:, 5]  # scores
                b = bbox[:, :5].clone()
                b[:, :2] += c
                # 0.3 could get results close to the ones from official merging script, even slightly better.
                i = ops.nms_rotated(b, scores, 0.3)
                bbox = bbox[i]

                b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
                for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                    classname = self.names[int(x[-1])].replace(" ", "-")
                    p = [round(i, 3) for i in x[:-2]]  # poly
                    score = round(x[-2], 3)

                    with open(f'{pred_merged_txt / f"Task1_{classname}"}.txt', "a") as f:
                        f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")

        return stats