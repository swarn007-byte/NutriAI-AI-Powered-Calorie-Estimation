import torch
import torch.nn as nn
import torch.nn.functional as F


class PortionNetLoss(nn.Module):
    def __init__(self, num_classes=108, lambda_cls=1.0, lambda_reg=0.1, lambda_distill=0.5,
                 label_smoothing=0.05, huber_delta=0.5, distill_temperature=4.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.lambda_distill = lambda_distill
        self.temperature = distill_temperature
        
        self.cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.l1_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss(delta=huber_delta)
    
    def classification_loss(self, logits, labels):
        return self.cls_loss_fn(logits, labels)
    
    def regression_loss(self, volume_pred, volume_true, energy_pred, energy_true):
        vol_l1 = self.l1_loss(volume_pred, volume_true)
        vol_huber = self.huber_loss(volume_pred, volume_true)
        vol_loss = vol_l1 + vol_huber
        
        energy_pred_norm = (energy_pred - energy_pred.mean()) / (energy_pred.std() + 1e-8)
        energy_true_norm = (energy_true - energy_true.mean()) / (energy_true.std() + 1e-8)
        
        eng_l1 = self.l1_loss(energy_pred_norm, energy_true_norm)
        eng_huber = self.huber_loss(energy_pred_norm, energy_true_norm)
        eng_loss = eng_l1 + eng_huber
        
        return 0.4 * vol_loss + 0.6 * eng_loss
    
    def distillation_loss(self, adapter_feat, pointnet_feat):
        mse_loss = F.mse_loss(adapter_feat, pointnet_feat)
        cosine_sim = F.cosine_similarity(adapter_feat, pointnet_feat, dim=1).mean()
        cosine_loss = 1 - cosine_sim
        
        adapter_prob = F.softmax(adapter_feat / self.temperature, dim=1)
        pointnet_prob = F.softmax(pointnet_feat / self.temperature, dim=1)
        kl_loss = F.kl_div(adapter_prob.log(), pointnet_prob, reduction='batchmean')
        
        return 0.7 * mse_loss + 0.2 * cosine_loss + 0.1 * kl_loss
    
    def forward(self, outputs, labels, volume_true, energy_true, mode='multimodal'):
        cls_loss = self.classification_loss(outputs['class_logits'], labels)
        reg_loss = self.regression_loss(outputs['volume'], volume_true, outputs['energy'], energy_true)
        
        if mode == 'multimodal' and outputs.get('pc_features_real') is not None:
            distill_loss = self.distillation_loss(outputs['pc_features_adapter'], outputs['pc_features_real'])
        else:
            distill_loss = torch.tensor(0.0, device=cls_loss.device)
        
        total_loss = self.lambda_cls * cls_loss + self.lambda_reg * reg_loss + self.lambda_distill * distill_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'distill_loss': distill_loss
        }
