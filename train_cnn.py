"""
CNN 基线模型 - CIFAR-10 图像分类
使用简单卷积神经网络进行分类
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report
import psutil
import os


# ============================================
# 第一部分：数据加载与预处理
# ============================================
print("=" * 60)
print("第一步：加载 CIFAR-10 数据集")
print("=" * 60)

# 数据转换：转换为 Tensor 并进行归一化
# CIFAR-10 的均值和标准差
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 从指定路径加载 CIFAR-10 训练集和测试集
train_dataset = torchvision.datasets.CIFAR10(
    root='D:/ML_Data/cifar10',
    train=True,
    download=False,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='D:/ML_Data/cifar10',
    train=False,
    download=False,
    transform=transform
)

print(f"原始训练集大小: {len(train_dataset)}")
print(f"原始测试集大小: {len(test_dataset)}")

# 随机降采样以确保与 SVM 对比实验的一致性
np.random.seed(42)
train_indices = np.random.choice(len(train_dataset), 5000, replace=False)
test_indices = np.random.choice(len(test_dataset), 1000, replace=False)

train_subset = torch.utils.data.Subset(train_dataset, train_indices)
test_subset = torch.utils.data.Subset(test_dataset, test_indices)

print(f"降采样后训练集大小: {len(train_subset)}")
print(f"降采样后测试集大小: {len(test_subset)}")

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(
    train_subset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_subset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)


# ============================================
# 第二部分：CNN 模型定义
# ============================================
print("\n" + "=" * 60)
print("第二步：构建 CNN 模型")
print("=" * 60)


class SimpleCNN(nn.Module):
    """
    简单的 CNN 模型
    包含 3 个卷积层和 2 个全连接层
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # 第一个卷积块：Conv -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积块：Conv -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三个卷积块：Conv -> ReLU -> MaxPool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        # 经过 3 次池化，32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 第一个卷积块
        x = self.pool1(self.relu1(self.conv1(x)))

        # 第二个卷积块
        x = self.pool2(self.relu2(self.conv2(x)))

        # 第三个卷积块
        x = self.pool3(self.relu3(self.conv3(x)))

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# 检测硬件：CPU 或 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建模型并移动到设备
model = SimpleCNN(num_classes=10).to(device)
print("模型结构:")
print(model)

# 计算模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params}")
print(f"可训练参数量: {trainable_params}")


# ============================================
# 第三部分：模型训练
# ============================================
print("\n" + "=" * 60)
print("第三步：训练 CNN 模型")
print("=" * 60)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 记录训练开始时间和内存
train_start_time = time.time()
process = psutil.Process(os.getpid())

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(device)
    mem_before_train = torch.cuda.memory_allocated(device) / 1024 / 1024
else:
    mem_before_train = process.memory_info().rss / 1024 / 1024

# 训练参数
num_epochs = 5

print(f"开始训练 {num_epochs} 个 epochs...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 移动数据到设备
        inputs, targets = inputs.to(device), targets.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch [{epoch + 1}/{num_epochs}] "
                  f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    print(f"Epoch [{epoch + 1}/{num_epochs}] "
          f"Loss: {epoch_loss:.4f} "
          f"Acc: {epoch_acc:.2f}%")

# 记录训练结束时间和内存
train_end_time = time.time()

if torch.cuda.is_available():
    mem_after_train = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    mem_peak = torch.cuda.max_memory_reserved(device) / 1024 / 1024
else:
    mem_after_train = process.memory_info().rss / 1024 / 1024

train_time = train_end_time - train_start_time

if torch.cuda.is_available():
    train_mem_usage = mem_peak - mem_before_train
else:
    train_mem_usage = mem_after_train - mem_before_train

print(f"\nCNN 训练总耗时: {train_time:.2f} 秒")
if torch.cuda.is_available():
    print(f"GPU 显存峰值: {mem_peak:.2f} MB")
else:
    print(f"CPU 内存消耗: {train_mem_usage:.2f} MB")


# ============================================
# 第四部分：模型评估
# ============================================
print("\n" + "=" * 60)
print("第四步：模型评估")
print("=" * 60)

# 在测试集上进行预测
model.eval()
all_predictions = []
all_targets = []

print("正在测试集上进行预测...")
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.numpy())

all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

# 计算准确率
accuracy = accuracy_score(all_targets, all_predictions)
print(f"\n测试集准确率 (Accuracy): {accuracy:.4f} ({accuracy * 100:.2f}%)")

# 打印详细分类报告
print("\n分类报告 (Classification Report):")
print(classification_report(
    all_targets,
    all_predictions,
    target_names=[
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
))


# ============================================
# 第五部分：复杂度分析总结
# ============================================
print("\n" + "=" * 60)
print("复杂度分析总结")
print("=" * 60)
print(f"CNN 训练耗时: {train_time:.2f} 秒")

if torch.cuda.is_available():
    print(f"GPU 显存峰值: {mem_peak:.2f} MB")
else:
    print(f"CPU 内存峰值变化: {train_mem_usage:.2f} MB")

print("\n" + "=" * 60)
print("程序执行完成！")
print("=" * 60)