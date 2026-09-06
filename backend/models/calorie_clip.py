"""
CalorieCLIP: Accurate Food Calorie Estimation from Images

Usage:
    from calorie_clip import CalorieCLIP
    
    model = CalorieCLIP.from_pretrained("HaploLLC/CalorieCLIP")
    calories = model.predict("food_image.jpg")
    print(f"Estimated: {calories:.0f} calories")
"""
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import json

try:
    import open_clip
except ImportError:
    raise ImportError("Please install open_clip: pip install open-clip-torch")


class RegressionHead(nn.Module):
    """Regression head for calorie prediction (matches training architecture)"""
    def __init__(self, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)


class CalorieCLIP(nn.Module):
    """
    CalorieCLIP: CLIP-based calorie estimation model
    
    Fine-tuned on Nutrition5k dataset with:
    - MAE: 54.3 calories
    - 60.7% predictions within 50 calories
    - 81.5% predictions within 100 calories
    """
    
    def __init__(self, clip_model, preprocess, regression_head):
        super().__init__()
        self.clip = clip_model
        self.preprocess = preprocess
        self.head = regression_head
        self.device = "cpu"
    
    @classmethod
    def from_pretrained(cls, model_path, device="cpu"):
        """Load CalorieCLIP from saved weights"""
        model_path = Path(model_path)
        
        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {"base_model": "ViT-B-32", "pretrained": "openai"}
        
        # Load CLIP
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            config.get("base_model", "ViT-B-32"),
            pretrained=config.get("pretrained", "openai")
        )
        
        # Create regression head
        head = RegressionHead(input_dim=512)
        
        # Load weights
        weights_path = model_path / "calorie_clip.pt"
        if not weights_path.exists():
            weights_path = model_path / "best_model.pt"
        
        if weights_path.exists():
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            
            # Load CLIP encoder weights
            if "clip_state" in checkpoint:
                clip_model.load_state_dict(checkpoint["clip_state"], strict=False)
            
            # Load regression head weights
            if "regressor_state" in checkpoint:
                head.load_state_dict(checkpoint["regressor_state"])
            elif "head_state" in checkpoint:
                head.load_state_dict(checkpoint["head_state"])
        
        model = cls(clip_model, preprocess, head)
        model.to(device)
        model.device = device
        model.eval()
        
        return model
    
    def encode_image(self, image):
        """Encode image to CLIP features"""
        with torch.no_grad():
            features = self.clip.encode_image(image).float()
            # Note: Do NOT normalize features - training didn't use normalization
        return features
    
    def forward(self, image):
        """Forward pass: image tensor -> calorie prediction"""
        features = self.encode_image(image)
        calories = self.head(features)
        return calories.squeeze(-1)
    
    def predict(self, image_path, return_features=False):
        """
        Predict calories from an image path or PIL Image
        
        Args:
            image_path: Path to image or PIL Image
            return_features: If True, also return CLIP features
            
        Returns:
            Estimated calories (float)
        """
        # Load and preprocess image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")
        
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            features = self.encode_image(image_tensor)
            calories = self.head(features).item()
        
        if return_features:
            return calories, features.cpu().numpy()
        return calories
    
    def predict_batch(self, images):
        """Predict calories for a batch of images"""
        tensors = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            tensors.append(self.preprocess(img))
        
        batch = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            features = self.encode_image(batch)
            calories = self.head(features).squeeze(-1)
        
        return calories.cpu().numpy()


# Convenience function
def load_model(model_path=".", device="cpu"):
    """Load CalorieCLIP model"""
    return CalorieCLIP.from_pretrained(model_path, device=device)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python calorie_clip.py <image_path>")
        print("       python calorie_clip.py <image1> <image2> ...")
        sys.exit(1)
    
    # Load model
    model = CalorieCLIP.from_pretrained(".")
    
    # Predict
    for img_path in sys.argv[1:]:
        calories = model.predict(img_path)
        print(f"{Path(img_path).name}: {calories:.0f} calories")
