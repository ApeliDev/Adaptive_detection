from transformers import ViTModel, ViTImageProcessor
import torch
import numpy as np

class ViTEncoder:
    def __init__(self, model_name='google/vit-base-patch16-224'):
        self.model = ViTModel.from_pretrained(model_name)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def encode(self, image):
        """Encode image into visual features"""
        if isinstance(image, np.ndarray):
            inputs = self.processor(images=image, return_tensors="pt")
        else:
            inputs = image
        
        with torch.no_grad():
            outputs = self.model(**inputs.to(self.device))
        
        # Return [CLS] token
        return outputs.last_hidden_state[:, 0, :]