import torch
import cv2
import numpy as np
import time
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from typing import Dict, List, Tuple

class SSDDetector:
    def __init__(self, config: Dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize SSD model
        self.model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        self.model.to(self.device)
        self.model.eval()
        
        # Set confidence threshold
        self.conf_thresh = config.get('conf_thresh', 0.5)
        
        # Warmup model
        self.warmup()
    
    def warmup(self):
        """Warmup model with dummy input"""
        dummy = torch.randn(1, 3, 300, 300).to(self.device)
        _ = self.model(dummy)
    
    def detect(self, image: np.ndarray, latency_budget: float = 33.3) -> Dict:
        """
        Run SSD detection
        Args:
            image: Input image (H, W, 3) in BGR format
            latency_budget: Time budget in milliseconds (not used for SSD)
        Returns:
            Dictionary containing detections and metrics
        """
        # Preprocess
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run detection
        start_time = time.time()
        
        with torch.no_grad():
            # Preprocess image
            input_tensor = self.preprocess(img_rgb)
            
            # Run inference
            predictions = self.model(input_tensor)
            
            # Postprocess results
            detections = self.postprocess(predictions[0], img_rgb.shape[:2])
        
        elapsed = (time.time() - start_time) * 1000
        
        return {
            "detections": detections,
            "metrics": {
                "inference_time": elapsed,
                "num_detections": len(detections)
            }
        }
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for SSD"""
        # Convert to tensor and normalize
        img = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)
        return img
    
    def postprocess(self, predictions: Dict, orig_shape: Tuple[int, int]) -> List[Dict]:
        """Postprocess SSD outputs"""
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        # Filter by confidence
        mask = scores > self.conf_thresh
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
                'class_name': self.model.config.id2label[int(label)],
                'source': 'ssd'
            })
        
        return detections 