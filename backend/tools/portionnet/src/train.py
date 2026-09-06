import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from tqdm import tqdm
import json

from models import PortionNet
from dataset import get_dataloaders
from losses import PortionNetLoss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_mape(y_true, y_pred, epsilon=1e-8):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, rgb_only_ratio=0.3):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        rgb = batch[0].to(device)
        pointcloud = batch[1].to(device)
        labels = batch[2].to(device)
        nutrition = batch[3].to(device)
        volume = nutrition[:, 0]
        energy = nutrition[:, 1]
        
        use_rgb_only = random.random() < rgb_only_ratio
        mode = 'rgb_only' if use_rgb_only else 'multimodal'
        
        if use_rgb_only:
            outputs = model(rgb, pointcloud=None, mode='rgb_only', return_features=False)
        else:
            outputs = model(rgb, pointcloud=pointcloud, mode='multimodal', return_features=True)
        
        loss_dict = criterion(outputs, labels, volume, energy, mode=mode)
        loss = loss_dict['total_loss']
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return {'total_loss': total_loss / len(train_loader)}


def validate(model, val_loader, criterion, device, mode='rgb_only'):
    model.eval()
    all_labels, all_preds = [], []
    all_volume_true, all_volume_pred = [], []
    all_energy_true, all_energy_pred = [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Validation ({mode})'):
            rgb = batch[0].to(device)
            pointcloud = batch[1].to(device) if mode == 'multimodal' else None
            labels = batch[2].to(device)
            nutrition = batch[3].to(device)
            volume = nutrition[:, 0]
            energy = nutrition[:, 1]
            
            outputs = model(rgb, pointcloud=pointcloud, mode=mode, return_features=False)
            loss_dict = criterion(outputs, labels, volume, energy, mode=mode)
            total_loss += loss_dict['total_loss'].item()
            
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
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy,
        'volume_mae': volume_mae,
        'volume_mape': volume_mape,
        'energy_mae': energy_mae,
        'energy_mape': energy_mape,
        'r2': r2
    }


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        root_dir=args.data_dir,
        excel_path=args.excel_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_pointcloud=True,
        train_split=0.8
    )
    
    print("Creating model...")
    model = PortionNet(num_classes=args.num_classes, feature_dim=args.feature_dim, num_heads=args.num_heads).to(device)
    
    criterion = PortionNetLoss(
        num_classes=args.num_classes,
        lambda_cls=args.lambda_cls,
        lambda_reg=args.lambda_reg,
        lambda_distill=args.lambda_distill
    )
    
    optimizer = optim.AdamW([
        {'params': model.rgb_encoder.parameters(), 'lr': args.lr_backbone},
        {'params': model.pointnet.parameters(), 'lr': args.lr_backbone},
        {'params': model.adapter.parameters(), 'lr': args.lr_head},
        {'params': model.fusion.parameters(), 'lr': args.lr_head},
        {'params': model.heads.parameters(), 'lr': args.lr_head}
    ], weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.lr_backbone, args.lr_backbone, args.lr_head, args.lr_head, args.lr_head],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_rgb_metrics': [], 'val_multimodal_metrics': []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, rgb_only_ratio=args.rgb_only_ratio)
        print(f"Train Loss: {train_metrics['total_loss']:.4f}")
        
        val_rgb_metrics = validate(model, val_loader, criterion, device, mode='rgb_only')
        val_multimodal_metrics = validate(model, val_loader, criterion, device, mode='multimodal')
        
        print(f"Val RGB - Acc: {val_rgb_metrics['accuracy']:.2f}%, Vol MAPE: {val_rgb_metrics['volume_mape']:.2f}%, Eng MAPE: {val_rgb_metrics['energy_mape']:.2f}%")
        print(f"Val Multimodal - Acc: {val_multimodal_metrics['accuracy']:.2f}%, Vol MAPE: {val_multimodal_metrics['volume_mape']:.2f}%, Eng MAPE: {val_multimodal_metrics['energy_mape']:.2f}%")
        
        history['train_loss'].append(train_metrics)
        history['val_rgb_metrics'].append(val_rgb_metrics)
        history['val_multimodal_metrics'].append(val_multimodal_metrics)
        
        if val_rgb_metrics['loss'] < best_val_loss:
            best_val_loss = val_rgb_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rgb_metrics': val_rgb_metrics,
                'val_multimodal_metrics': val_multimodal_metrics
            }, os.path.join(args.output_dir, f'best_model_seed{args.seed}.pt'))
            print("✓ Saved best model")
    
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, os.path.join(args.output_dir, f'final_model_seed{args.seed}.pt'))
    
    with open(os.path.join(args.output_dir, f'history_seed{args.seed}.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PortionNet')
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--excel_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--num_classes', type=int, default=108)
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_backbone', type=float, default=1e-4)
    parser.add_argument('--lr_head', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lambda_cls', type=float, default=1.0)
    parser.add_argument('--lambda_reg', type=float, default=0.1)
    parser.add_argument('--lambda_distill', type=float, default=0.5)
    parser.add_argument('--rgb_only_ratio', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=7)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
