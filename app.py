import torch
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn


class DualPathwayObjectDetection(nn.Module):
    def __init__(self, num_classes, num_anchors=7):
        super(DualPathwayObjectDetection, self).__init__()
        
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.output_channels_yolo = num_anchors * (num_classes + 5)
        self.output_channels_ssd = num_anchors * (num_classes + 4)
        
        self.yolo_pathway = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.output_channels_yolo, kernel_size=1)
        )
        
        self.ssd_pathway = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.output_channels_ssd, kernel_size=1)
        )
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        if x.size(1) != 3:
            x = x.permute(0, 3, 1, 2)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Get pathway outputs
        yolo_output = self.yolo_pathway(features)
        ssd_output = self.ssd_pathway(features)
        
        # Get dimensions
        batch_size = x.size(0)
        H, W = yolo_output.size(2), yolo_output.size(3)
        
        # Reshape YOLO output with proper dimension calculation
        yolo_output = yolo_output.permute(0, 2, 3, 1).contiguous()
        yolo_output = yolo_output.view(batch_size, H, W, self.num_anchors, self.num_classes + 5)
        yolo_output = yolo_output.permute(0, 3, 4, 1, 2)  # [B, A, C+5, H, W]
        
        # Extract components
        box_coords = yolo_output[:, :, :4]  # [B, A, 4, H, W]
        objectness = yolo_output[:, :, 4:5]  # [B, A, 1, H, W]
        class_pred = yolo_output[:, :, 5:]  # [B, A, C, H, W]
        
        # Concatenate for final output
        output = torch.cat([box_coords, objectness, class_pred], dim=2).requires_grad_()
        
        return output
    
def compute_loss(self, predictions, targets, boxes, valid_boxes_mask):
    """
    Compute the combined loss for object detection
    Args:
        predictions: shape [batch_size, num_anchors, (4 + 1 + num_classes), H, W]
        targets: shape [batch_size, max_boxes]
        boxes: shape [batch_size, max_boxes, 4]
        valid_boxes_mask: shape [batch_size, max_boxes]
    """
    batch_size = predictions.size(0)
    
    # Extract components from predictions
    pred_boxes = predictions[:, :, :4]  # [batch_size, num_anchors, 4, H, W]
    pred_obj = predictions[:, :, 4]     # [batch_size, num_anchors, H, W]
    pred_cls = predictions[:, :, 5:]    # [batch_size, num_anchors, num_classes, H, W]
    
    # Initialize loss components
    box_loss = torch.tensor(0.0, device=self.device)
    obj_loss = torch.tensor(0.0, device=self.device)
    cls_loss = torch.tensor(0.0, device=self.device)
    total_valid_boxes = 0
    
    # Convert valid_boxes_mask to boolean tensor
    valid_boxes_mask = valid_boxes_mask.bool()
    
    # Process each item in batch
    for b in range(batch_size):
        # Get valid boxes for this batch item
        valid_mask = valid_boxes_mask[b]
        valid_boxes = boxes[b][valid_mask]
        valid_targets = targets[b][valid_mask]
        
        if valid_boxes.numel() == 0:
            continue
        
        # Reshape predictions for this batch item
        batch_pred_boxes = pred_boxes[b].permute(0, 2, 3, 1).contiguous().view(-1, 4)
        batch_pred_obj = pred_obj[b].contiguous().view(-1)
        batch_pred_cls = pred_cls[b].permute(0, 2, 3, 1).contiguous().view(-1, pred_cls.size(2))
        
        try:
            # Compute IoU between predictions and ground truth boxes
            ious = box_iou(batch_pred_boxes, valid_boxes)  # [num_anchors*H*W, num_valid_boxes]
            
            # For each ground truth box, find the best matching prediction
            best_ious, best_n = ious.max(dim=0)  # [num_valid_boxes]
            
            # Compute box regression loss
            box_loss += F.mse_loss(
                batch_pred_boxes[best_n],
                valid_boxes,
                reduction='sum'
            )
            
            # Compute objectness loss
            obj_targets = torch.zeros_like(batch_pred_obj)
            obj_targets[best_n] = 1.0
            obj_loss += F.binary_cross_entropy_with_logits(
                batch_pred_obj,
                obj_targets,
                reduction='sum'
            )
            
            # Compute classification loss
            cls_loss += F.cross_entropy(
                batch_pred_cls[best_n],
                valid_targets,
                reduction='sum'
            )
            
            total_valid_boxes += len(valid_boxes)
            
        except RuntimeError as e:
            print(f"Error in batch {b}:")
            print(f"Pred boxes shape: {batch_pred_boxes.shape}")
            print(f"Valid boxes shape: {valid_boxes.shape}")
            print(f"Valid targets shape: {valid_targets.shape}")
            raise e
    
    # Normalize losses
    eps = 1e-6
    if total_valid_boxes > 0:
        box_loss = box_loss / total_valid_boxes
        obj_loss = obj_loss / total_valid_boxes
        cls_loss = cls_loss / total_valid_boxes
    else:
        box_loss = box_loss * 0
        obj_loss = obj_loss * 0
        cls_loss = cls_loss * 0
    
    # Combine losses with weights
    total_loss = box_loss * 5.0 + obj_loss * 1.0 + cls_loss * 1.0
    
    return total_loss, {
        'box_loss': box_loss.item(),
        'obj_loss': obj_loss.item(),
        'cls_loss': cls_loss.item(),
        'total_loss': total_loss.item()
    }

def box_iou(box1, box2):
    """
    Compute IoU between two sets of boxes
    Args:
        box1: [N, 4] in xywh format
        box2: [M, 4] in xywh format
    Returns:
        IoU matrix of shape [N, M]
    """
    # Convert xywh to xyxy
    b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
    b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
    b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2
    
    # Get intersection rectangle
    x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    
    # Intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = b1_area.unsqueeze(1) + b2_area - intersection
    
    # IoU
    iou = intersection / (union + 1e-16)
    return iou
class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.1, verbose=True
        )
    
    def compute_loss(self, predictions, targets, boxes, valid_boxes_mask):
        """
        Compute the combined loss for object detection
        Args:
            predictions: shape [batch_size, num_anchors, (4 + 1 + num_classes), H, W]
            targets: shape [batch_size, max_boxes]
            boxes: shape [batch_size, max_boxes, 4]
            valid_boxes_mask: shape [batch_size, max_boxes]
        """
        batch_size = predictions.size(0)
        
        # Extract components from predictions
        pred_boxes = predictions[:, :, :4]  # [batch_size, num_anchors, 4, H, W]
        pred_obj = predictions[:, :, 4]     # [batch_size, num_anchors, H, W]
        pred_cls = predictions[:, :, 5:]    # [batch_size, num_anchors, num_classes, H, W]
        
        # Initialize loss components
        box_loss = torch.tensor(0.0, device=self.device)
        obj_loss = torch.tensor(0.0, device=self.device)
        cls_loss = torch.tensor(0.0, device=self.device)
        total_valid_boxes = 0
        
        # Process each item in batch
        for b in range(batch_size):
            # Get valid boxes for this batch item
            valid_mask = valid_boxes_mask[b]
            valid_boxes = boxes[b, valid_mask]
            valid_targets = targets[b, valid_mask]
            
            if len(valid_boxes) == 0:
                continue
            
            # Reshape predictions for this batch item
            # [num_anchors, 4, H, W] -> [num_anchors * H * W, 4]
            batch_pred_boxes = pred_boxes[b].permute(0, 2, 3, 1).reshape(-1, 4)
            # [num_anchors, H, W] -> [num_anchors * H * W]
            batch_pred_obj = pred_obj[b].reshape(-1)
            # [num_anchors, num_classes, H, W] -> [num_anchors * H * W, num_classes]
            batch_pred_cls = pred_cls[b].permute(0, 2, 3, 1).reshape(-1, pred_cls.size(2))
            
            # Compute IoU between predictions and ground truth boxes
            ious = box_iou(batch_pred_boxes, valid_boxes)  # [num_anchors*H*W, num_valid_boxes]
            
            # For each ground truth box, find the best matching prediction
            best_ious, best_n = ious.max(dim=0)  # [num_valid_boxes]
            
            # Compute box regression loss
            box_loss += F.mse_loss(
                batch_pred_boxes[best_n],
                valid_boxes,
                reduction='sum'
            )
            
            # Compute objectness loss
            obj_targets = torch.zeros_like(batch_pred_obj)
            obj_targets[best_n] = 1.0
            obj_loss += F.binary_cross_entropy_with_logits(
                batch_pred_obj,
                obj_targets,
                reduction='sum'
            )
            
            # Compute classification loss
            cls_loss += F.cross_entropy(
                batch_pred_cls[best_n],
                valid_targets,
                reduction='sum'
            )
            
            total_valid_boxes += len(valid_boxes)
        
        # Normalize losses
        eps = 1e-6
        box_loss = box_loss / (total_valid_boxes + eps)
        obj_loss = obj_loss / (total_valid_boxes + eps)
        cls_loss = cls_loss / (total_valid_boxes + eps)
        
        # Combine losses with weights
        total_loss = box_loss * 5.0 + obj_loss * 1.0 + cls_loss * 1.0
        
        return total_loss, {
            'box_loss': box_loss.item(),
            'obj_loss': obj_loss.item(),
            'cls_loss': cls_loss.item(),
            'total_loss': total_loss.item()
        }

    def box_iou(box1, box2):
        """
        Compute IoU between two sets of boxes
        Args:
            box1: [N, 4] in xywh format
            box2: [M, 4] in xywh format
        Returns:
            IoU matrix of shape [N, M]
        """
        # Convert xywh to xyxy
        b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
        b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
        b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2
        
        # Get intersection rectangle
        x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
        y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
        x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
        y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
        
        # Intersection area
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union = b1_area.unsqueeze(1) + b2_area - intersection
        
        # IoU
        iou = intersection / (union + 1e-16)
        return iou

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch_idx, (images, boxes, targets, valid_boxes_mask) in enumerate(pbar):
                images = images.to(self.device)
                boxes = boxes.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(images)
                
                loss, loss_dict = self.compute_loss(predictions, targets, boxes, valid_boxes_mask)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })
        
        return total_loss / num_batches
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation') as pbar:
                for batch_idx, (images, boxes, targets, valid_boxes_mask) in enumerate(pbar):
                    images = images.to(self.device)
                    boxes = boxes.to(self.device)
                    targets = targets.to(self.device)
                    
                    predictions = self.model(images).requires_grad_()
                    
                    loss, loss_dict = self.compute_loss(predictions, targets, boxes, valid_boxes_mask)
                    
                    total_loss += loss.item()
                    pbar.set_postfix({
                        'val_loss': f'{loss.item():.4f}',
                        'avg_val_loss': f'{total_loss / (batch_idx + 1):.4f}'
                    })
        
        return total_loss / num_batches
    
    def train(self, num_epochs):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, 'best_model.pth')
            
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')

         
def load_yaml(file_path='data.yaml'):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / 'images'
        self.label_dir = self.root_dir / 'labels'
        self.transform = transform
        self.image_files = list(self.image_dir.glob('*.jpg')) + list(self.image_dir.glob('*.png'))
        
        self.config = load_yaml()
        self.class_names = self.config['names']
        self.num_classes = len(self.class_names)
    
    def __len__(self):
        return len(self.image_files)
    
    def parse_label_line(self, line):
        try:
            data = line.strip().split()
            if len(data) != 5:
                return None, None
            
            class_id = int(data[0])
            coords = list(map(float, data[1:]))
            
            if not all(0 <= x <= 1 for x in coords):
                return None, None
                
            return class_id, coords
        except (ValueError, IndexError):
            return None, None
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original dimensions
        height, width = image.shape[:2]
        
        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, coords = self.parse_label_line(line)
                    if class_id is not None and coords is not None:
                        if 0 <= class_id < self.num_classes:
                            boxes.append(coords)
                            labels.append(class_id)
        
        # Convert to tensors before transform
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long)
        
        if self.transform:
            try:
                # Apply transforms
                transformed = self.transform(image=image, bboxes=boxes.numpy(), class_labels=labels.numpy())
                image = transformed['image']  # Now a torch tensor [C, H, W]
                
                if len(transformed['bboxes']) > 0:
                    boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                    labels = torch.tensor(transformed['class_labels'], dtype=torch.long)
                else:
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros(0, dtype=torch.long)
            except Exception as e:
                print(f"Transform failed for image {img_path}: {str(e)}")
                # Return empty tensors if transform fails
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.long)
                # Ensure image is properly transformed even if bbox transform fails
                image = self.transform(image=image)['image']
        
        return image, boxes, labels

def custom_collate_fn(batch):
    images = []
    boxes = []
    labels = []
    
    for image, box, label in batch:
        images.append(image)
        boxes.append(box)
        labels.append(label)
    
    images = torch.stack(images, dim=0)
    
    max_boxes = max(box.shape[0] for box in boxes)
    padded_boxes = []
    padded_labels = []
    valid_boxes_mask = []
    
    for box, label in zip(boxes, labels):
        num_boxes = box.shape[0]
        if num_boxes < max_boxes:
            pad_boxes = torch.zeros((max_boxes - num_boxes, 4), dtype=torch.float32, device=box.device)
            box = torch.cat([box, pad_boxes], dim=0)
            
            pad_labels = torch.zeros(max_boxes - num_boxes, dtype=torch.long, device=label.device)
            label = torch.cat([label, pad_labels], dim=0)
            
            mask = torch.cat([torch.ones(num_boxes, dtype=torch.bool), torch.zeros(max_boxes - num_boxes, dtype=torch.bool)])
        else:
            mask = torch.ones(num_boxes, dtype=torch.bool)
        
        padded_boxes.append(box)
        padded_labels.append(label)
        valid_boxes_mask.append(mask)
    
    padded_boxes = torch.stack(padded_boxes, dim=0)
    padded_labels = torch.stack(padded_labels, dim=0)
    valid_boxes_mask = torch.stack(valid_boxes_mask, dim=0)
    
    return images, padded_boxes, padded_labels, valid_boxes_mask

def compute_output_shape(model):
    """
    Compute the output shape of the model using a dummy input
    Args:
        model: The DualPathwayObjectDetection model
    Returns:
        tuple: The shape of the model's output
    """
    # Move model to CPU for shape computation
    model = model.cpu()
    
    # Create a dummy input with batch size 1
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # Put model in eval mode
    model.eval()
    
    with torch.no_grad():
        try:
            # Get output shape
            output = model(dummy_input)
            return output.shape
        except Exception as e:
            print(f"Error computing output shape: {str(e)}")
            return None
        finally:
            # Reset model to training mode
            model.train()

def main():
    config = load_yaml()
    num_classes = len(config['names'])
    
    BATCH_SIZE = 8
    NUM_EPOCHS = 5 # here i di 5 coz while doing 50  is was throwing error while on 10 sooee...
    IMAGE_SIZE = 640
    
    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    val_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    train_dataset = CustomDataset('train', transform=train_transform)
    val_dataset = CustomDataset('valid', transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualPathwayObjectDetection(num_classes=num_classes, num_anchors=7)
    
    output_shape = compute_output_shape(model)
    print(f"Model output shape: {output_shape}")
    
    model = model.to(device)
    
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.train(NUM_EPOCHS)

if __name__ == '__main__':
    main()
