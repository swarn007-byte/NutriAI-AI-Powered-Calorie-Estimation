import argparse
import torch
from PIL import Image

from models import PortionNet
from dataset import get_transforms


def load_model(checkpoint_path, device='cuda'):
    model = PortionNet(num_classes=108, feature_dim=256, num_heads=8)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def predict(model, image_path, device='cuda'):
    transform = get_transforms(augment=False)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor, pointcloud=None, mode='rgb_only', return_features=False)
    
    class_idx = torch.argmax(outputs['class_logits'], dim=1).item()
    class_prob = torch.softmax(outputs['class_logits'], dim=1)[0, class_idx].item()
    volume = outputs['volume'].item()
    energy = outputs['energy'].item()
    
    return {
        'food_class': f'Class {class_idx}',
        'confidence': class_prob * 100,
        'volume_ml': volume,
        'energy_kcal': energy
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully!")
    
    print(f"\nAnalyzing image: {args.image}")
    results = predict(model, args.image, device)
    
    print("\nNUTRITION ESTIMATION RESULTS")
    print(f"Food Item:     {results['food_class']}")
    print(f"Confidence:    {results['confidence']:.2f}%")
    print(f"Volume:        {results['volume_ml']:.2f} mL")
    print(f"Energy:        {results['energy_kcal']:.2f} kcal")
    
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PortionNet Inference')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    main(args)
