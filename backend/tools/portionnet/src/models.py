import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, resnet18


class PointNet(nn.Module):
    def __init__(self, num_points=1024, feature_dim=256):
        super().__init__()
        self.num_points = num_points
        
        self.scale_embedding = nn.Linear(3, 64)
        
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        
        self.pool_scales = [64, 128, 256, 512]
        self.multi_pool = nn.ModuleList([nn.AdaptiveMaxPool1d(scale) for scale in self.pool_scales])
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        scale_feat_dim = 4800
        self.fc = nn.Sequential(
            nn.Linear(scale_feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(1)
        
        if num_points < 64:
            padding = torch.zeros(batch_size, 64 - num_points, 3, device=x.device)
            x = torch.cat([x, padding], dim=1)
            num_points = 64
        
        x_transposed = x.transpose(1, 2)
        
        scale_factors = torch.stack([x.max(dim=1)[0] - x.min(dim=1)[0]], dim=1).view(batch_size, 3)
        scale_features = self.scale_embedding(scale_factors)
        
        x1 = self.conv1(x_transposed)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        
        multi_res_features = []
        for i, pool in enumerate(self.multi_pool[:2]):
            pooled = pool(x5)
            pooled = self.global_max_pool(pooled).squeeze(-1)
            multi_res_features.append(pooled)
        
        x5_max = self.global_max_pool(x5).squeeze(-1)
        x5_avg = self.global_avg_pool(x5).squeeze(-1)
        x4_max = self.global_max_pool(x4).squeeze(-1)
        x3_max = self.global_max_pool(x3).squeeze(-1)
        
        x3_max_reduced = x3_max[:, :128]
        
        all_features = [x5_max, x5_avg, x4_max, x3_max_reduced, scale_features] + multi_res_features
        multi_scale_features = torch.cat(all_features, dim=1)
        
        features = self.fc(multi_scale_features)
        return features


class DualRGBEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        
        vit = vit_b_16(weights='IMAGENET1K_V1')
        vit.heads = nn.Identity()
        self.vit = vit
        
        resnet = resnet18(weights='IMAGENET1K_V1')
        resnet.fc = nn.Identity()
        self.resnet = resnet
        
        self.projection = nn.Sequential(
            nn.Linear(768 + 512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, feature_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        vit_feat = self.vit(x)
        resnet_feat = self.resnet(x)
        combined = torch.cat([vit_feat, resnet_feat], dim=1)
        return self.projection(combined)


class RGBToPointCloudAdapter(nn.Module):
    def __init__(self, input_dim=256, output_dim=256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.adapter(x)


class CrossModalAttention(nn.Module):
    def __init__(self, rgb_dim, pc_dim, hidden_dim=256):
        super().__init__()
        self.rgb_proj = nn.Linear(rgb_dim, hidden_dim)
        self.pc_proj = nn.Linear(pc_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, rgb_feat, pc_feat):
        rgb_proj = self.rgb_proj(rgb_feat).unsqueeze(1)
        pc_proj = self.pc_proj(pc_feat).unsqueeze(1)
        
        rgb_attended, _ = self.attention(rgb_proj, pc_proj, pc_proj)
        rgb_attended = self.norm1(rgb_attended + rgb_proj)
        
        pc_attended, _ = self.attention(pc_proj, rgb_proj, rgb_proj)
        pc_attended = self.norm2(pc_attended + pc_proj)
        
        return rgb_attended.squeeze(1), pc_attended.squeeze(1)


class FusionModule(nn.Module):
    def __init__(self, feature_dim=256, fusion_dim=512):
        super().__init__()
        self.cross_attention = CrossModalAttention(feature_dim, feature_dim, feature_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU()
        )
    
    def forward(self, rgb_feat, pc_feat):
        rgb_attended, pc_attended = self.cross_attention(rgb_feat, pc_feat)
        combined = torch.cat([rgb_attended, pc_attended], dim=1)
        return self.fusion(combined)


class PredictionHeads(nn.Module):
    def __init__(self, feature_dim=256, num_classes=108):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        self.volume_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 1),
            nn.Softplus()
        )
        
        self.energy_head = nn.Sequential(
            nn.Linear(feature_dim + 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()
        )
    
    def forward(self, features):
        class_logits = self.classifier(features)
        volume = self.volume_head(features)
        
        energy_input = torch.cat([features, volume], dim=1)
        energy = self.energy_head(energy_input)
        
        return {
            'class_logits': class_logits,
            'volume': volume.squeeze(-1),
            'energy': energy.squeeze(-1)
        }


class PortionNet(nn.Module):
    def __init__(self, num_classes=108, feature_dim=256, num_heads=8):
        super().__init__()
        
        self.rgb_encoder = DualRGBEncoder(feature_dim=feature_dim)
        self.pointnet = PointNet(feature_dim=feature_dim)
        self.adapter = RGBToPointCloudAdapter(input_dim=feature_dim, output_dim=feature_dim)
        self.fusion = FusionModule(feature_dim=feature_dim, fusion_dim=512)
        self.heads = PredictionHeads(feature_dim=256, num_classes=num_classes)
    
    def forward(self, rgb, pointcloud=None, mode='rgb_only', return_features=False):
        rgb_feat = self.rgb_encoder(rgb)
        
        pc_features_real = None
        pc_features_adapter = None
        
        if mode == 'multimodal' and pointcloud is not None:
            pc_feat = self.pointnet(pointcloud)
            pc_features_real = pc_feat
            if self.training and return_features:
                pc_features_adapter = self.adapter(rgb_feat)
        else:
            pc_feat = self.adapter(rgb_feat)
            pc_features_adapter = pc_feat
        
        fused_feat = self.fusion(rgb_feat, pc_feat)
        predictions = self.heads(fused_feat)
        
        if return_features:
            return {
                **predictions,
                'rgb_features': rgb_feat,
                'pc_features_real': pc_features_real,
                'pc_features_adapter': pc_features_adapter,
                'fused_features': fused_feat
            }
        else:
            return predictions
