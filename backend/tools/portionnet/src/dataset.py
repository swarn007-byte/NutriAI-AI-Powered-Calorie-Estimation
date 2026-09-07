import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


class FoodRGBPointCloud(Dataset):
    def __init__(self, root_dir, excel_path, transform_rgb=None, use_pointcloud=True):
        self.root_dir = root_dir
        self.transform_rgb = transform_rgb
        self.use_pointcloud = use_pointcloud
        
        df = pd.read_excel(excel_path)
        df.columns = df.columns.str.strip()
        
        self.metadata = {}
        for _, row in df.iterrows():
            obj_name = row["Object_name"].strip()
            nutrition = np.array([row["Volume"], row["Energy (Kcal)"]], dtype=np.float32)
            self.metadata[obj_name] = nutrition
        
        self.original_metadata = self.metadata.copy()
        for k, v in self.metadata.items():
            log_transformed = np.log1p(v)
            self.metadata[k] = torch.tensor(log_transformed, dtype=torch.float32)
        
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            for dirpath, dirnames, filenames in os.walk(class_dir):
                if "Original" in dirnames:
                    rgb_dir = os.path.join(dirpath, "Original")
                    try:
                        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        for rgb_file in rgb_files:
                            rgb_path = os.path.join(rgb_dir, rgb_file)
                            if os.path.exists(rgb_path):
                                nutrition = self.metadata.get(cls, torch.zeros(2))
                                self.samples.append((rgb_path, self.class_to_idx[cls], cls))
                    except Exception:
                        continue
    
    def load_pointcloud(self, obj_name, sample_idx, pointcloud_root, num_points=1024):
        if not self.use_pointcloud or not OPEN3D_AVAILABLE:
            return None
        
        try:
            pointcloud_dir = os.path.join(pointcloud_root, obj_name)
            if not os.path.exists(pointcloud_dir):
                return None
            
            ply_files = sorted([f for f in os.listdir(pointcloud_dir) if f.endswith('.ply')])
            if not ply_files:
                return None
            
            ply_file = ply_files[sample_idx % len(ply_files)]
            ply_path = os.path.join(pointcloud_dir, ply_file)
            
            pcd = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(pcd.points)
            
            if len(points) == 0:
                return None
            
            centroid = np.mean(points, axis=0)
            points = points - centroid
            max_dist = np.max(np.linalg.norm(points, axis=1))
            if max_dist > 0:
                points = points / max_dist
            
            if len(points) > num_points:
                indices = np.random.choice(len(points), num_points, replace=False)
                points = points[indices]
            elif len(points) < num_points:
                repeat_factor = num_points // len(points) + 1
                points = np.tile(points, (repeat_factor, 1))[:num_points]
            
            return torch.tensor(points, dtype=torch.float32)
        except Exception:
            return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        rgb_path, label, obj_name = self.samples[idx]
        
        try:
            rgb_img = Image.open(rgb_path).convert("RGB")
            if self.transform_rgb:
                rgb_img = self.transform_rgb(rgb_img)
            
            pointcloud = self.load_pointcloud(obj_name, idx, "/path/to/pointclouds")
            if pointcloud is None:
                pointcloud = torch.randn(1024, 3) * 0.1
            
            nutrition = self.metadata.get(obj_name, torch.zeros(2))
            return rgb_img, pointcloud, label, nutrition
        except Exception:
            dummy_rgb = torch.zeros(3, 224, 224)
            dummy_pointcloud = torch.randn(1024, 3) * 0.1
            dummy_nutrition = torch.zeros(2)
            return dummy_rgb, dummy_pointcloud, 0, dummy_nutrition


def get_transforms(augment=True):
    if augment:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_dataloaders(root_dir, excel_path, batch_size=16, num_workers=4, use_pointcloud=True, train_split=0.8):
    full_dataset = FoodRGBPointCloud(
        root_dir=root_dir,
        excel_path=excel_path,
        transform_rgb=get_transforms(augment=False),
        use_pointcloud=use_pointcloud
    )
    
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_dataset.dataset.transform_rgb = get_transforms(augment=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader
