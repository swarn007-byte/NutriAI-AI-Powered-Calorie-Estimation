import os
import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from tqdm import tqdm
import json

from models import PortionNet
from dataset import get_dataloaders


def compute_mape(y_true, y_pred, epsilon=1e-8):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def evaluate(model, data_loader, device, mode='rgb_only'):
    model.eval()
    all_labels, all_preds = [], []
    all_volume_true, all_volume_pred = [], []
    all_energy_true, all_energy_pred = [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Evaluating ({mode})'):
            rgb = batch[0].to(device)
            pointcloud = batch[1].to(device) if mode == 'multimodal' else None
            labels = batch[2].to(device)
            nutrition = batch[3].to(device)
            volume = nutrition[:, 0]
            energy = nutrition[:, 1]
            
            outputs = model(rgb, pointcloud=pointcloud, mode=mode, return_features=False)
            preds = torch.argmax(outputs['class_logits'], dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_volume_true.extend(volume.cpu().numpy())
            all_volume_pred.extend(outputs['volume'].cpu().numpy())
            all_energy_true.extend(energy.cpu().numpy())
            all_energy_pred.extend(outputs['energy'].cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    volume_mae = mean_absolute_error(all_volume_true, all_volume_pred)
    volume_mape = compute_mape(all_volume_true, all_volume_pred)
    energy_mae = mean_absolute_error(all_energy_true, all_energy_pred)
    energy_mape = compute_mape(all_energy_true, all_energy_pred)
    r2 = r2_score(all_volume_true, all_volume_pred)
    
    return {
        'accuracy': accuracy,
        'volume_mae': volume_mae,
        'volume_mape': volume_mape,
        'energy_mae': energy_mae,
        'energy_mape': energy_mape,
        'r2': r2
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading model...")
    model = PortionNet(num_classes=args.num_classes, feature_dim=args.feature_dim, num_heads=args.num_heads).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    print("Loading dataset...")
    _, test_loader = get_dataloaders(
        root_dir=args.data_dir,
        excel_path=args.excel_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_pointcloud=args.mode == 'multimodal',
        train_split=0.8
    )
    
    print(f"\nEvaluating in {args.mode} mode...")
    metrics = evaluate(model, test_loader, device, mode=args.mode)
    
    print(f"\nEVALUATION RESULTS ({args.mode.upper()} MODE)")
    print(f"Classification Accuracy: {metrics['accuracy']:.2f}%")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"Volume MAE: {metrics['volume_mae']:.2f} mL, MAPE: {metrics['volume_mape']:.2f}%")
    print(f"Energy MAE: {metrics['energy_mae']:.2f} kcal, MAPE: {metrics['energy_mape']:.2f}%")
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PortionNet')
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--excel_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=108)
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--mode', type=str, default='rgb_only', choices=['rgb_only', 'multimodal'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_file', type=str, default=None)
    
    args = parser.parse_args()
    main(args)
