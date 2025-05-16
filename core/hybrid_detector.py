import torch
import cv2
import numpy as np
import time
import torchvision
from transformers import DetrForObjectDetection, DetrImageProcessor
from yolov9.models.yolo import Model as YOLOv9
from typing import Dict, List, Tuple
from yolov9.utils.general import non_max_suppression

class HybridDetector:
    def __init__(self, config: Dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize YOLOv9
        self.yolo = YOLOv9(config['yolo']['weights']).to(self.device)
        self.yolo.conf = config['yolo']['conf_thresh']
        self.yolo.iou = config['yolo']['iou_thresh']
        
        # Initialize DETR
        self.detr = DetrForObjectDetection.from_pretrained(
            config['detr']['model']
        ).to(self.device)
        self.detr_processor = DetrImageProcessor.from_pretrained(config['detr']['model'])
        self.detr_conf_thresh = config['detr']['conf_thresh']
        
        # Warmup models
        self.warmup()
    
    def warmup(self):
        """Warmup models with dummy input"""
        dummy = torch.randn(1, 3, 640, 640).to(self.device)
        _ = self.yolo(dummy)
        _ = self.detr(dummy)
    
    def detect(self, image: np.ndarray, latency_budget: float = 33.3) -> Dict:
        """
        Run hybrid detection with adaptive computation
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
            # YOLO detection
            yolo_input = self.preprocess_yolo(img_rgb)
            yolo_out = self.yolo(yolo_input)
            yolo_dets = self.postprocess_yolo(yolo_out)
            
            # DETR detection if within budget
            elapsed = (time.time() - start_time) * 1000
            if elapsed < latency_budget * 0.7:  # Use 70% of budget for DETR
                detr_input = self.preprocess_detr(img_rgb)
                detr_out = self.detr(**detr_input)
                detr_dets = self.postprocess_detr(detr_out, img_rgb.shape[:2])
            else:
                detr_dets = []
        
        # Combine results
        combined = self.ensemble_detections(yolo_dets, detr_dets)
        
        return {
            "detections": combined,
            "metrics": {
                "yolo_time": elapsed,
                "detr_time": (time.time() - start_time) * 1000 - elapsed,
                "num_detections": len(combined)
            }
        }
    
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
    
    def preprocess_detr(self, image: np.ndarray) -> Dict:
        """Preprocess image for DETR"""
        if isinstance(image, np.ndarray):
            inputs = self.detr_processor(images=image, return_tensors="pt")
        else:
            inputs = {"pixel_values": image}  # Ensure inputs is a dict
        return inputs.to(self.device)
    
    def postprocess_detr(self, outputs, orig_shape) -> List[Dict]:
        """Postprocess DETR outputs"""
        target_sizes = torch.tensor([orig_shape]).to(self.device)
        results = self.detr_processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes,
            threshold=self.detr_conf_thresh
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append({
                'bbox': box.tolist(),
                'confidence': score.item(),
                'class_id': label.item(),
                'class_name': self.detr.config.id2label[label.item()],
                'source': 'detr'
            })
        return detections
    
    def ensemble_detections(self, yolo_dets: List[Dict], detr_dets: List[Dict]) -> List[Dict]:
        """Combine detections from both models"""
        # Simple non-maximum suppression between models
        all_dets = yolo_dets + detr_dets
        if not all_dets:
            return []
        
        # Convert to tensors for processing
        boxes = torch.tensor([d['bbox'] for d in all_dets])
        scores = torch.tensor([d['confidence'] for d in all_dets])
        
        # Apply NMS across model outputs
        keep = torchvision.ops.nms(boxes, scores, 0.5)
        
        return [all_dets[i] for i in keep.tolist()]