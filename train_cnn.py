"""
CNN 基线模型 - CIFAR-10 图像分类
使用简单卷积神经网络进行分类
增加样本量 + 多次实验取平均
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
# 配置参数
# ============================================
TRAIN_SAMPLES = 10000
TEST_SAMPLES = 1000
NUM_EXPERIMENTS = 3
NUM_EPOCHS = 5
BATCH_SIZE = 64

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

print("=" * 60)
print("CNN 实验配置")
print("=" * 60)
print(f"训练集样本数: {TRAIN_SAMPLES}")
print(f"测试集样本数: {TEST_SAMPLES}")
print(f"重复实验次数: {NUM_EXPERIMENTS}")
print(f"训练轮数: {NUM_EPOCHS}")
print(f"批次大小: {BATCH_SIZE}")


# ============================================
# 第一部分：数据加载
# ============================================
print("\n" + "=" * 60)
print("第一步：加载 CIFAR-10 数据集")
print("=" * 60)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

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


# ============================================
# CNN 模型定义
# ============================================
class SimpleCNN(nn.Module):
    """简单的 CNN 模型 - 3 个卷积层 + 2 个全连接层"""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ============================================
# 重复实验主循环
# ============================================
all_accuracies = []
all_train_times = []
all_train_mems = []

for exp_id in range(NUM_EXPERIMENTS):
    print("\n" + "=" * 60)
    print(f"第 {exp_id + 1}/{NUM_EXPERIMENTS} 次实验")
    print("=" * 60)

    random_seed = 42 + exp_id * 100
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    train_indices = np.random.choice(len(train_dataset), TRAIN_SAMPLES, replace=False)
    test_indices = np.random.choice(len(test_dataset), TEST_SAMPLES, replace=False)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)

    print(f"随机种子: {random_seed}")
    print(f"训练集样本数: {len(train_subset)}")
    print(f"测试集样本数: {len(test_subset)}")

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # 第二部分：模型构建
    print("\n" + "-" * 40)
    print("第二步：构建 CNN 模型")
    print("-" * 40)

    model = SimpleCNN(num_classes=10).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params}")

    # 第三部分：模型训练
    print("\n" + "-" * 40)
    print("第三步：训练 CNN 模型")
    print("-" * 40)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    process = psutil.Process(os.getpid())

    train_start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        mem_before_train = torch.cuda.memory_allocated(device) / 1024 / 1024
    else:
        mem_before_train = process.memory_info().rss / 1024 / 1024

    print(f"开始训练 {NUM_EPOCHS} 个 epochs...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 30 == 0:
                print(f"  Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                      f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
              f"Loss: {epoch_loss:.4f} "
              f"Acc: {epoch_acc:.2f}%")

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

    # 第四部分：模型评估
    print("\n" + "-" * 40)
    print("第四步：模型评估")
    print("-" * 40)

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

    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"\n测试集准确率 (Accuracy): {accuracy:.4f} ({accuracy * 100:.2f}%)")

    if exp_id == NUM_EXPERIMENTS - 1:
        print("\n分类报告 (Classification Report):")
        print(classification_report(
            all_targets,
            all_predictions,
            target_names=[
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
        ))

    all_accuracies.append(accuracy * 100)
    all_train_times.append(train_time)
    all_train_mems.append(train_mem_usage)

    print(f"\n本次实验: 耗时 {train_time:.2f} 秒, 内存 {train_mem_usage:.2f} MB")


# ============================================
# 第五部分：统计分析
# ============================================
print("\n" + "=" * 60)
print("统计结果汇总")
print("=" * 60)

mean_accuracy = np.mean(all_accuracies)
std_accuracy = np.std(all_accuracies)
mean_train_time = np.mean(all_train_times)
mean_train_mem = np.mean(all_train_mems)

print(f"\n准确率 (Accuracy):")
print(f"  各次实验: {[f'{a:.2f}%' for a in all_accuracies]}")
print(f"  平均值 ± 标准差: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")

print(f"\n训练耗时 (秒):")
print(f"  各次实验: {[f'{t:.2f}' for t in all_train_times]}")
print(f"  平均值: {mean_train_time:.2f} 秒")

print(f"\n内存消耗 (MB):")
print(f"  各次实验: {[f'{m:.2f}' for m in all_train_mems]}")
print(f"  平均值: {mean_train_mem:.2f} MB")

print("\n" + "=" * 60)
print("程序执行完成！")
print("=" * 60)