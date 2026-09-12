# 📋 UniVIP垃圾分类 - 完整执行方案（最终版）

## 🎯 项目概览

| 项目 | 详情 |
|------|------|
| **项目名称** | UniVIP-Trash-Classification |
| **核心思路** | 实现CVPR 2022的UniVIP自监督学习框架用于垃圾分类 |
| **数据集** | TrashNet (6类垃圾: cardboard, glass, metal, paper, plastic, trash) |
| **预期准确率** | 88-92% |
| **项目周期** | 3-4周 |
| **代码规模** | ~800行 (精简版) |
| **主要成果** | 完整代码 + 训练模型 + 对比分析 + 学位论文 |

---

## 🏗️ 项目整体架构

### **核心思路流程图**

```
┌─────────────────────────────────────────────────────────────────┐
│                     UniVIP完整训练流程                            │
└─────────────────────────────────────────────────────────────────┘

【阶段1: 数据准备】
  TrashNet原始数据
      ↓
  [train/val/test分割 7:1:2]
      ↓
  数据增强
  ├─ View1: 重增强 (对比学习)
  │  └─ RandomCrop + ColorJitter + GaussianBlur
  └─ View2: 轻增强 (重建)
     └─ Crop + Flip

【阶段2: 模型架构】
  View1, View2
      ↓
  ┌─ViT骨干网络 (768维特征)
  │  ├─ 对比学习分支
  │  │  ├─ CLS token提取
  │  │  └─ 对比头 (MLP: 768→128)
  │  │     ↓
  │  │  对比特征 [B, 128]
  │  │
  │  └─ 遮挡重建分支
  │     ├─ Patch tokens提取
  │     ├─ 应用mask (75%)
  │     └─ 重建头 (MLP: 768→768)
  │        ↓
  │     重建特征 [B, 768]

【阶段3: 损失函数】
  对比损失 (NT-Xent)
       +
  重建损失 (MSE)
       ↓
  联合损失 = λ₁×L_contrast + λ₂×L_reconstruct

【阶段4A: 预训练】
  输入: 无标签图片 (100 epochs)
  优化: 联合损失
  输出: 预训练好的ViT骨干
  
【阶段4B: 微调】
  输入: 有标签垃圾分类数据 (30 epochs)
  骨干: 冻结
  优化: 分类头
  输出: 垃圾分类模型

【阶段5: 评估】
  测试集准确率 + 混淆矩阵
  对比: UniVIP vs ResNet50 vs CLIP
  分析: 自监督学习的优势
```

---

## 📁 项目文件结构

```
UniVIP-Trash-Classification/
│
├── 📁 data/                                # 数据目录
│   ├── raw/                                # 原始TrashNet数据
│   │   ├── cardboard/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   └── trash/
│   │
│   └── processed/                          # 处理后的数据
│       ├── train/                          # 70%训练数据
│       │   ├── cardboard/
│       │   ├── glass/
│       │   └── ...
│       ├── val/                            # 10%验证数据
│       │   └── ...
│       └── test/                           # 20%测试数据
│           └── ...
│
├── 📁 src/                                 # 源代码
│   ├── model.py                            # 【阶段2】模型架构
│   │   ├─ ViTBackbone
│   │   ├─ ContrastiveHead
│   │   ├─ ReconstructionHead
│   │   └�� UniVIPModel
│   │
│   ├── losses.py                           # 【阶段3】损失函数
│   │   ├─ ContrastiveLoss (NT-Xent)
│   │   ├─ ReconstructionLoss (MSE)
│   │   └─ UnifiedLoss
│   │
│   ├── dataset.py                          # 【阶段1】数据加载
│   │   ├─ view1_transform (重增强)
│   │   ├─ view2_transform (轻增强)
│   │   └─ DualViewDataset
│   │
│   ├── train.py                            # 【阶段4A】预训练循环
│   │   └─ pretrain_univip()
│   │
│   ├── finetune.py                         # 【阶段4B】微调循环
│   │   └─ finetune_classifier()
│   │
│   └── utils.py                            # 【阶段5】评估和可视化
│       ├─ evaluate_model()
│       ├─ plot_confusion_matrix()
│       └─ compare_methods()
│
├── 📁 results/                             # 结果输出
│   ├── models/
│   │   ├── univip_backbone_final.pth       # 预训练模型
│   │   └── best_classifier.pth             # 微调模型
│   │
│   ├── logs/
│   │   ├── pretrain.log
│   │   └── finetune.log
│   │
│   └── metrics/
│       ├── pretrain_curves.png
│       ├── finetune_curves.png
│       ├── confusion_matrix.png
│       └── results.json
│
├── 📄 config.yaml                          # 配置文件
├── 📄 run_all.py                           # 一键运行脚本
├── 📄 requirements.txt                     # 依赖列表
├── 📄 README.md                            # 项目说明
│
├── 📁 notebooks/                           # Jupyter笔记本
│   ├── 01_data_exploration.ipynb           # 数据探索
│   └── 02_results_analysis.ipynb           # 结果分析
│
└── 📁 paper/                               # 论文相关
    ├── 课程设计报告.md
    ├── UniVIP方法说明.md
    └── 对比分析.md
```

---

## 🔧 核心代码框架详解

### **【阶段1】数据处理** (`src/dataset.py`)

#### 数据增强策略

**对比学习视图 (View1) - 重增强:**
```python
view1_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),           # 随机裁剪缩放
    transforms.RandomHorizontalFlip(p=0.5),      # 水平翻转
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # 颜色抖动
    transforms.RandomApply([GaussianBlur()], p=0.5),  # 高斯模糊
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])
])
```

**重建视图 (View2) - 轻增强:**
```python
view2_transform = transforms.Compose([
    transforms.Resize(256),                      # 缩放
    transforms.CenterCrop(224),                  # 中心裁剪
    transforms.RandomHorizontalFlip(p=0.3),      # 轻微翻转
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])
])
```

#### DualViewDataset类

```python
class DualViewDataset(Dataset):
    """为每个图片生成两个不同的视图"""
    
    def __init__(self, root, transform1, transform2):
        self.images = [...]  # 所有图片路径
        self.transform1 = transform1  # 对比学习视图
        self.transform2 = transform2  # 重建视图
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        view1 = self.transform1(img)   # [3, 224, 224]
        view2 = self.transform2(img)   # [3, 224, 224]
        label = self.get_label(idx)    # 类别标签
        return view1, view2, label
```

---

### **【阶段2】模型架构** (`src/model.py`)

#### ViT骨干网络

```python
class ViTBackbone(nn.Module):
    """Vision Transformer骨干 - 特征提取器"""
    
    def __init__(self, model_name='vit_base_patch16_224'):
        super().__init__()
        # 使用timm库加载预训练ViT
        self.vit = timm.create_model(
            model_name,
            pretrained=True,      # ImageNet预训练
            num_classes=0,        # 移除分类层
            global_pool=''        # 保留所有tokens
        )
        self.embed_dim = 768  # ViT-B的特征维度
    
    def forward(self, x):
        """
        输入:  x [B, 3, 224, 224]
        输出: features [B, 197, 768]
               197 = 196个patch + 1个CLS token
        """
        features = self.vit.forward_features(x)
        return features  # [B, 197, 768]
```

#### 对比学习头

```python
class ContrastiveHead(nn.Module):
    """将768维特征投影到128维对比空间"""
    
    def __init__(self, input_dim=768, output_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, output_dim)
        )
    
    def forward(self, x):
        """
        输入:  x [B, 768]  (CLS token)
        输出: [B, 128]     (对比特征)
        """
        return self.head(x)
```

#### 重建头

```python
class ReconstructionHead(nn.Module):
    """重建被遮挡的patch"""
    
    def __init__(self, input_dim=768, output_dim=768):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_dim)
        )
    
    def forward(self, x):
        """
        输入:  x [B, 768]  (patch tokens)
        输出: [B, 768]    (重建的patch)
        """
        return self.head(x)
```

#### 完整UniVIP模型

```python
class UniVIPModel(nn.Module):
    """
    UniVIP: 统一的自监督学习框架
    同时做对比学习和遮挡图像建模
    """
    
    def __init__(self, mask_ratio=0.75):
        super().__init__()
        self.backbone = ViTBackbone()
        self.contrast_head = ContrastiveHead()
        self.reconstruction_head = ReconstructionHead()
        self.mask_ratio = mask_ratio  # 75%的patch被遮挡
    
    def forward(self, view1, view2):
        """
        前向传播
        
        输入:
          view1: [B, 3, 224, 224] - 对比学习视图
          view2: [B, 3, 224, 224] - 重建视图
        
        输出:
          {
            'contrast1': [B, 128]         - view1的对比特征
            'contrast2': [B, 128]         - view2的对比特征
            'reconstructed': [B*mask, 768] - 重建的patch
            'original_patches': [B*mask, 768] - 原始patch
          }
        """
        
        # ===== 对比学习分支 =====
        feat1 = self.backbone(view1)        # [B, 197, 768]
        feat2 = self.backbone(view2)        # [B, 197, 768]
        
        # 提取CLS token (第一个token)
        cls1 = feat1[:, 0, :]               # [B, 768]
        cls2 = feat2[:, 0, :]               # [B, 768]
        
        # 投影到对比空间
        contrast1 = self.contrast_head(cls1)  # [B, 128]
        contrast2 = self.contrast_head(cls2)  # [B, 128]
        
        # ===== 遮挡重建分支 =====
        patches2 = feat2[:, 1:, :]          # [B, 196, 768] (去掉CLS)
        
        # 创建随机遮挡mask
        B, N, D = patches2.shape
        mask = torch.rand(B, N) < self.mask_ratio  # [B, 196]
        mask = mask.to(patches2.device)
        
        # 提取被遮挡的patches
        masked_patches = patches2[mask]          # [B*mask_count, 768]
        original_patches = patches2[mask]        # 原始patches
        
        # 重建
        reconstructed = self.reconstruction_head(masked_patches)
        
        return {
            'contrast1': contrast1,
            'contrast2': contrast2,
            'reconstructed': reconstructed,
            'original_patches': original_patches,
            'mask': mask
        }
```

---

### **【阶段3】损失函数** (`src/losses.py`)

#### 对比损失 (NT-Xent Loss)

```python
class ContrastiveLoss(nn.Module):
    """
    NT-Xent 对比损失
    来自SimCLR论文，是自监督学习的标准方法
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, feat1, feat2):
        """
        让同一图片的两个视图特征接近，
        不同图片的特征远离
        
        输入:
          feat1: [B, 128] - 第一个视图的对比特征
          feat2: [B, 128] - 第二个视图的对比特征
        
        输出:
          loss: 标量
        """
        
        # 1. 特征归一化 (重要!)
        feat1 = F.normalize(feat1, dim=1)  # [B, 128]
        feat2 = F.normalize(feat2, dim=1)  # [B, 128]
        
        # 2. 计算相似度矩阵
        similarity = torch.matmul(feat1, feat2.T) / self.temperature  # [B, B]
        
        # 3. 正样本标签 (对角线)
        labels = torch.arange(feat1.shape[0], device=feat1.device)
        
        # 4. NT-Xent损失 (cross entropy)
        loss = F.cross_entropy(similarity, labels)
        
        return loss
```

**直观理解:**
```
相似度矩阵 (温度=0.07后):
       对应view2
          ↓
对应   [ 高  低  低  低 ]  ← 同一图片对应位置
view1 [ 低  高  低  低 ]     应该都是高
      [ 低  低  高  低 ]
      [ 低  低  低  高 ]

目标: 对角线都是1, 其他位置都是0
      用cross_entropy来优化这个目标
```

#### 重建损失 (MSE Loss)

```python
class ReconstructionLoss(nn.Module):
    """
    L2重建损失
    让模型能够重建被遮挡的patches
    """
    
    def forward(self, reconstructed, original):
        """
        输入:
          reconstructed: [N, 768] - 重建的patches
          original:      [N, 768] - 原始patches
        
        输出:
          loss: 标量
        """
        loss = F.mse_loss(reconstructed, original)
        return loss
```

#### 联合损失

```python
class UniVIPLoss(nn.Module):
    """
    联合损失函数
    = λ₁ × 对比损失 + λ₂ × 重建损失
    """
    
    def __init__(self, lambda_contrast=1.0, lambda_reconstruct=0.5):
        super().__init__()
        self.contrast_loss = ContrastiveLoss()
        self.reconstruct_loss = ReconstructionLoss()
        self.lambda_c = lambda_contrast
        self.lambda_r = lambda_reconstruct
    
    def forward(self, outputs):
        """
        输入: model的输出字典
        
        输出: 
          {
            'total': 总损失,
            'contrast': 对比损失,
            'reconstruct': 重建损失
          }
        """
        
        # 计算两个损失
        loss_contrast = self.contrast_loss(
            outputs['contrast1'],
            outputs['contrast2']
        )
        
        loss_reconstruct = self.reconstruct_loss(
            outputs['reconstructed'],
            outputs['original_patches']
        )
        
        # 加权求和
        total_loss = (self.lambda_c * loss_contrast + 
                     self.lambda_r * loss_reconstruct)
        
        return {
            'total': total_loss,
            'contrast': loss_contrast,
            'reconstruct': loss_reconstruct
        }
```

---

### **【阶段4A】预训练循环** (`src/train.py`)

```python
def pretrain_univip(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    epochs=100,
    device='cuda',
    save_dir='./results/models'
):
    """
    UniVIP预训练循环
    
    输入: 无标签图片数据
    目标: 学习好的特征表示
    """
    
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        
        total_loss = 0
        loss_c_total = 0
        loss_r_total = 0
        
        for batch_idx, (view1, view2, _) in enumerate(train_loader):
            # 1. 数据到GPU
            view1 = view1.to(device)
            view2 = view2.to(device)
            
            # 2. 前向传播
            outputs = model(view1, view2)
            
            # 3. 计算损失
            losses = criterion(outputs)
            loss = losses['total']
            
            # 4. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 5. 记录损失
            total_loss += loss.item()
            loss_c_total += losses['contrast'].item()
            loss_r_total += losses['reconstruct'].item()
        
        # 学习率衰减
        scheduler.step()
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        avg_loss_c = loss_c_total / len(train_loader)
        avg_loss_r = loss_r_total / len(train_loader)
        
        # 输出日志
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Contrast Loss: {avg_loss_c:.4f}")
        print(f"  Reconstruct Loss: {avg_loss_r:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save(model.backbone.state_dict(), ckpt_path)
            print(f"  ✅ Checkpoint saved: {ckpt_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'univip_backbone_final.pth')
    torch.save(model.backbone.state_dict(), final_model_path)
    print(f"\n✅ 预训练完成！模型保存到: {final_model_path}")
    
    return final_model_path
```

---

### **【阶段4B】微调循环** (`src/finetune.py`)

```python
def finetune_classifier(
    backbone_path,
    train_loader,
    val_loader,
    epochs=30,
    device='cuda',
    save_dir='./results/models'
):
    """
    微调分类器
    
    输入: 有标签的垃圾分类数据
    目标: 学习分类决策
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 加载预训练的骨干网络
    backbone = ViTBackbone()
    backbone.load_state_dict(torch.load(backbone_path))
    backbone.to(device)
    backbone.eval()  # 冻结
    
    # 2. 冻结骨干网络参数
    for param in backbone.parameters():
        param.requires_grad = False
    
    # 3. 定义分类头
    classifier = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, 6)  # 6个垃圾类
    ).to(device)
    
    # 4. 定义优化器 (只优化分类头)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    
    best_val_acc = 0
    best_model_path = None
    
    for epoch in range(epochs):
        # ===== 训练 =====
        classifier.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, _, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # 特征提取 (不计算梯度)
            with torch.no_grad():
                features = backbone(images)      # [B, 197, 768]
                cls_token = features[:, 0, :]    # [B, 768]
            
            # 分类
            logits = classifier(cls_token)
            loss = criterion(logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = torch.max(logits, 1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # ===== 验证 =====
        classifier.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, _, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                features = backbone(images)
                cls_token = features[:, 0, :]
                logits = classifier(cls_token)
                
                _, pred = torch.max(logits, 1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        
        # 日志
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        
        # 保存最好的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(save_dir, 'best_classifier.pth')
            torch.save(classifier.state_dict(), best_model_path)
            print(f"  ✅ 最好的模型已保存 (Val Acc: {val_acc:.2f}%)")
    
    print(f"\n✅ 微调完成！最好模型准确率: {best_val_acc:.2f}%")
    
    return best_model_path
```

---

### **【阶段5】评估** (`src/utils.py`)

```python
def evaluate_on_test_set(
    backbone_path,
    classifier_path,
    test_loader,
    device='cuda'
):
    """
    在测试集上评估模型
    计算准确率、混淆矩阵、分类报告
    """
    
    # 加载模型
    backbone = ViTBackbone()
    backbone.load_state_dict(torch.load(backbone_path))
    backbone.to(device)
    backbone.eval()
    
    classifier = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, 6)
    ).to(device)
    classifier.load_state_dict(torch.load(classifier_path))
    classifier.eval()
    
    # 评估
    all_preds = []
    all_labels = []
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, _, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            features = backbone(images)
            cls_token = features[:, 0, :]
            logits = classifier(cls_token)
            
            _, pred = torch.max(logits, 1)
            test_correct += (pred == labels).sum().item()
            test_total += labels.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100 * test_correct / test_total
    
    print(f"\n✅ 测试集准确率: {test_acc:.2f}%")
    
    # 混淆矩阵
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵:")
    print(cm)
    
    # 分类报告
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return test_acc, cm
```

---

### **【一键脚本】** (`run_all.py`)

```python
"""
一键运行UniVIP垃圾分类全流程
"""

import argparse
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='UniVIP Trash Classification')
    parser.add_argument('--mode', 
                       choices=['prepare', 'pretrain', 'finetune', 'eval', 'all'],
                       default='all',
                       help='运行模式')
    parser.add_argument('--config', default='config.yaml', help='配置文件')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 【步骤1】数据准备
    if args.mode in ['prepare', 'all']:
        print("=" * 50)
        print("📊 数据准备中...")
        print("=" * 50)
        # 调用 prepare_data()
    
    # 【步骤2】预训练
    if args.mode in ['pretrain', 'all']:
        print("\n" + "=" * 50)
        print("🚀 预训练UniVIP...")
        print("=" * 50)
        # 调用 pretrain_univip()
    
    # 【步骤3】微调
    if args.mode in ['finetune', 'all']:
        print("\n" + "=" * 50)
        print("🎯 微调分类器...")
        print("=" * 50)
        # 调用 finetune_classifier()
    
    # 【步骤4】评估
    if args.mode in ['eval', 'all']:
        print("\n" + "=" * 50)
        print("📈 评估结果...")
        print("=" * 50)
        # 调用 evaluate_on_test_set()
    
    print("\n✅ 全部完成！")

if __name__ == '__main__':
    main()
```

---

## 📅 时间规划和工作流程

### **周期1: 框架搭建 (3-4天)**

```
Day 1-2: 数据处理 + 模型架构
  ├─ 实现 dataset.py
  ├─ 实现 model.py (3个模块)
  └─ 数据加载测试

Day 3: 损失函数 + 训练脚本
  ├─ 实现 losses.py (3个损失)
  ├─ 实现 train.py
  └─ 代码集成测试

Day 4: 微调 + 评估 + 一键脚本
  ├─ 实现 finetune.py
  ├─ 实现 utils.py
  └─ 实现 run_all.py
```

### **周期2: 模型训练 (5-7天)**

```
Day 5-7: 预训练 (100 epochs)
  ├─ 运行 python run_all.py --mode pretrain
  ├─ 监控损失曲线
  └─ 保存检查点

Day 8-10: 微调 (30 epochs)
  ├─ 运行 python run_all.py --mode finetune
  ├─ 优化超参
  └─ 获得最终模型

Day 11: 评估 + 对比
  ├─ 运行 python run_all.py --mode eval
  ├─ 生成对比表格
  └─ 收集所有结果
```

### **周期3: 报告和总结 (4-5天)**

```
Day 12-13: 数据分析
  ├─ 混淆矩阵分析
  ├─ 失败案例研究
  └─ 特征可视化

Day 14-15: 论文撰写
  ├─ 写方法部分 (UniVIP详细说明)
  ├─ 写实验部分 (结果表格)
  ├─ 写分析部分 (对比讨论)
  └─ 生成参考文献

Day 16: PPT制作 + 演讲准备
  ├─ 制作演示PPT
  ├─ 准备演讲稿
  └─ 模型演示准备
```

---

## 📊 配置文件 (`config.yaml`)

```yaml
# 数据配置
data:
  data_dir: './data/processed'
  batch_size: 64
  num_workers: 4
  num_classes: 6
  image_size: 224

# 预训练配置
pretrain:
  epochs: 100
  learning_rate: 1e-3
  weight_decay: 1e-4
  warmup_epochs: 5
  
  # UniVIP特定
  mask_ratio: 0.75              # 遮挡比例
  contrast_temperature: 0.07     # 对比学习温度
  lambda_contrast: 1.0           # 对比损失权重
  lambda_reconstruct: 0.5        # 重建损失权重

# 微调配置
finetune:
  epochs: 30
  learning_rate: 1e-4
  weight_decay: 0
  freeze_backbone: true          # 冻结ViT骨干

# 模型配置
model:
  backbone: 'vit_base_patch16_224'
  contrast_dim: 128
  classifier_hidden: 256

# 硬件和随机性
device: 'cuda'
seed: 42
```

---

## 🎯 预期结果

### **性能对比表**

| 方法 | 预训练 | 微调 | 测试准确率 | 优势 |
|------|--------|------|----------|------|
| **UniVIP** | ✅ 100 epochs | ✅ 30 epochs | **89-92%** | 数据高效 + 泛化强 |
| ResNet50 | ❌ | ✅ 50 epochs | 87-90% | 基础方案 |
| CLIP微调 | ❌ (预计算) | ✅ 30 epochs | 88-91% | 对标方案 |

### **对比维度**

1. **准确率对比**
   - UniVIP vs ResNet50: 期望提升2-5%
   - UniVIP vs CLIP: 可能相当或略优

2. **数据效率对比**
   - 用50%数据: UniVIP准确率下降<3%, ResNet下降>10%
   - 用10%数据: UniVIP仍可达85%, ResNet只有75%

3. **收敛速度对比**
   - UniVIP预训练: 100 epochs (~10小时)
   - ResNet微调: 50 epochs (~5小时)

---

## 📝 论文框架 (8000字)

```
【标题】
UniVIP框架在垃圾分类中的应用研究

【摘要】 (200字)
论文将CVPR 2022提出的UniVIP自监督学习框架应用于垃圾分类任务。
通过统一对比学习和遮挡图像建模，在TrashNet数据集上取得XX%的准确率，
相比传统监督学习提升了YY%，展现了自监督学习在数据高效性上的优势。

【第1章 引言】 (600字)
1.1 背景: 垃圾分类的重要性
1.2 问题: 传统方法需要大量标注数据
1.3 机遇: 自监督学习的发展
1.4 贡献: 首次应用UniVIP到垃圾分类

【第2章 相关工作】 (800字)
2.1 自监督学习综述
    2.1.1 对比学习 (SimCLR, MoCo)
    2.1.2 遮挡预测 (MAE, BEiT)
    2.1.3 混合方法 (UniVIP)
2.2 垃圾分类研究现状

【第3章 方法】 (1200字)
3.1 数据增强策略
    3.1.1 对比学习视图 (重增强)
    3.1.2 重建视图 (轻增强)
3.2 UniVIP模型架构
    3.2.1 ViT骨干网络
    3.2.2 对比学习分支
    3.2.3 遮挡重建分支
3.3 损失函数设计
    3.3.1 NT-Xent对比损失
    3.3.2 MSE重建损失
    3.3.3 联合损失函数
3.4 训练策略
    3.4.1 无标签预训练
    3.4.2 有标签微调

【第4章 实验】 (2000字)
4.1 数据集和设置
    4.1.1 TrashNet数据集 (6类, 总数X)
    4.1.2 数据划分 (7:1:2)
    4.1.3 实现细节
4.2 主要结果
    表4.1: 与基线方法的对比
    - UniVIP: XX%
    - ResNet50: YY%
    - EfficientNet: ZZ%
4.3 消融实验
    表4.2: 损失函数的贡献
    - 完整UniVIP: XX%
    - 仅对比学习: YY%
    - 仅重建学习: ZZ%
4.4 数据效率分析
    表4.3: 不同数据量下的性能
    - 100%数据: UniVIP XX%, ResNet YY%
    - 50%数据: UniVIP AA%, ResNet BB%
    - 10%数据: UniVIP CC%, ResNet DD%

【第5章 分析】 (1200字)
5.1 混淆矩阵分析
    图5.1: 混淆矩阵热力图
    - 哪些类容易混淆?
    - 为什么?
5.2 失败案例研究
    图5.2: 典型失败案例
    - cardboard vs paper
    - metal vs glass
5.3 特征表示分析
    图5.3: t-SNE可视化
    - 预训练前后的对比
    - 类别分离度

【第6章 结论】 (400字)
6.1 主要发现
6.2 自监督学习的优势
6.3 局限和未来工作

【参考文献】 (20+篇)
```

---

## ✅ 完整清单

### **代码实现**
- [ ] `src/dataset.py` - 数据加载和增强
- [ ] `src/model.py` - UniVIP模型架构
- [ ] `src/losses.py` - 三个损失函数
- [ ] `src/train.py` - 预训练循环
- [ ] `src/finetune.py` - 微调循环
- [ ] `src/utils.py` - 评估和可视化
- [ ] `run_all.py` - 一键脚本
- [ ] `config.yaml` - 配置文件

### **模型和结果**
- [ ] 预训练模型 (100 epochs)
- [ ] 微调模型 (30 epochs)
- [ ] 测试集准确率报告
- [ ] 混淆矩阵图
- [ ] 训练曲线图

### **论文和报告**
- [ ] 课程设计报告 (8000字)
- [ ] UniVIP方法说明文档
- [ ] 对比分析报告
- [ ] 演讲PPT (15-20分钟)

### **其他**
- [ ] GitHub仓库
- [ ] README.md
- [ ] requirements.txt
- [ ] Jupyter笔记本 (可选)

---

## 🚀 现在可以开始了！

这份方案包含了：
✅ 完整的核心代码框架
✅ 详细的实现细节
✅ 清晰的时间规划
✅ 预期的实验结果
✅ 论文写作框架

**下一步:** 启动Copilot Coding Agent按照这个方案严格执行！
