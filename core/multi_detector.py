import torch
import cv2
import numpy as np
import time
import torchvision
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'yolov9'))

from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from yolov9.models.yolo import Model as YOLOv9
from typing import Dict, List, Tuple
from yolov9.utils.general import non_max_suppression

class MultiDetector:
    def __init__(self, config: Dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize SSD model
        self.ssd = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        self.ssd.to(self.device)
        self.ssd.eval()
        self.ssd_conf_thresh = config['ssd']['conf_thresh']
        
        # Initialize YOLO model
        yolo_weights = os.path.join(os.path.dirname(__file__), '..', config['yolo']['weights'])
        self.yolo = YOLOv9(yolo_weights).to(self.device)
        self.yolo.conf = config['yolo']['conf_thresh']
        self.yolo.iou = config['yolo']['iou_thresh']
        self.yolo.eval()
        
        # Warmup models
        self.warmup()
    
    def warmup(self):
        """Warmup models with dummy input"""
        dummy = torch.randn(1, 3, 300, 300).to(self.device)
        _ = self.ssd(dummy)
        dummy = torch.randn(1, 3, 640, 640).to(self.device)
        _ = self.yolo(dummy)
    
    def detect(self, image: np.ndarray, latency_budget: float = 33.3) -> Dict:
        """
        Run detection with both SSD and YOLO
        Args:
            image: Input image (H, W, 3) in BGR format
            latency_budget: Time budget in milliseconds
        Returns:
            Dictionary containing detections and metrics
        """
        # Preprocess
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run both detectors in parallel if budget allows
        start_time = time.time()
        
        with torch.no_grad():
            # SSD detection
            ssd_input = self.preprocess_ssd(img_rgb)
            ssd_out = self.ssd(ssd_input)
            ssd_dets = self.postprocess_ssd(ssd_out[0], img_rgb.shape[:2])
            
            # YOLO detection if within budget
            elapsed = (time.time() - start_time) * 1000
            if elapsed < latency_budget * 0.7:  # Use 70% of budget for YOLO
                yolo_input = self.preprocess_yolo(img_rgb)
                yolo_out = self.yolo(yolo_input)
                yolo_dets = self.postprocess_yolo(yolo_out)
            else:
                yolo_dets = []
        
        # Combine results
        combined = self.ensemble_detections(ssd_dets, yolo_dets)
        
        return {
            "detections": combined,
            "metrics": {
                "ssd_time": elapsed,
                "yolo_time": (time.time() - start_time) * 1000 - elapsed,
                "num_detections": len(combined)
            }
        }
    
    def preprocess_ssd(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for SSD"""
        img = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)
        return img
    
    def postprocess_ssd(self, predictions: Dict, orig_shape: Tuple[int, int]) -> List[Dict]:
        """Postprocess SSD outputs"""
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        # Filter by confidence
        mask = scores > self.ssd_conf_thresh
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # Convert to list of detections
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            detections.append({
                'bbox': box.tolist(),
                'confidence': float(score),
                'class_id': int(label),
                'class_name': self.ssd.config.id2label[int(label)],
                'source': 'ssd'
            })
        
        return detections
    
    def preprocess_yolo(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for YOLO"""
        img = cv2.resize(image, (640, 640))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0
        return img.unsqueeze(0).to(self.device)
    
    def postprocess_yolo(self, preds: torch.Tensor) -> List[Dict]:
        """Postprocess YOLO outputs"""
        preds = non_max_suppression(preds[0], self.yolo.conf, self.yolo.iou)
        detections = []
        for det in preds[0]:  # First (and only) image in batch
            x1, y1, x2, y2, conf, cls = det.tolist()
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'class_id': int(cls),
                'class_name': self.yolo.names[int(cls)],
                'source': 'yolo'
            })
        return detections
    
    def ensemble_detections(self, ssd_dets: List[Dict], yolo_dets: List[Dict]) -> List[Dict]:
        """Combine detections from both models"""
        # Simple non-maximum suppression between models
        all_dets = ssd_dets + yolo_dets
        if not all_dets:
            return []
        
        # Convert to tensors for processing
        boxes = torch.tensor([d['bbox'] for d in all_dets])
        scores = torch.tensor([d['confidence'] for d in all_dets])
        
        # Apply NMS across model outputs
        keep = torchvision.ops.nms(boxes, scores, 0.5)
        
        return [all_dets[i] for i in keep.tolist()] 