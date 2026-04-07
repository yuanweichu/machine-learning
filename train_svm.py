"""
SVM 基线模型 - CIFAR-10 图像分类
使用 HOG 特征进行分类
"""

import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import psutil
import os


# ============================================
# 第一部分：数据加载与降采样
# ============================================
print("=" * 60)
print("第一步：加载 CIFAR-10 数据集")
print("=" * 60)

# 数据转换：将 PIL 图片转换为 numpy 数组
transform = transforms.Compose([
    transforms.ToTensor()
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

# 随机降采样以加速训练（防止 SVM 过慢）
np.random.seed(42)
train_indices = np.random.choice(len(train_dataset), 5000, replace=False)
test_indices = np.random.choice(len(test_dataset), 1000, replace=False)

train_subset = torch.utils.data.Subset(train_dataset, train_indices)
test_subset = torch.utils.data.Subset(test_dataset, test_indices)

print(f"降采样后训练集大小: {len(train_subset)}")
print(f"降采样后测试集大小: {len(test_subset)}")

# 提取数据和标签
train_images = [train_dataset[i][0].numpy() for i in train_indices]
train_labels = [train_dataset[i][1] for i in train_indices]

test_images = [test_dataset[i][0].numpy() for i in test_indices]
test_labels = [test_dataset[i][1] for i in test_indices]


# ============================================
# 第二部分：HOG 特征提取
# ============================================
print("\n" + "=" * 60)
print("第二步：提取 HOG 特征")
print("=" * 60)

# 记录特征提取开始时间和内存
feature_start_time = time.time()
process = psutil.Process(os.getpid())
mem_before_feature = process.memory_info().rss / 1024 / 1024  # MB


def extract_hog_features(image):
    """
    从单张图片中提取 HOG 特征
    输入: image (3, 32, 32) 的 torch tensor
    输出: HOG 特征向量
    """
    # 转换为 (32, 32, 3) 并转为 uint8
    img = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
    
    # 转换为灰度图
    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    
    # 提取 HOG 特征
    # pixels_per_cell: 每个 cell 的像素数
    # cells_per_block: 每个 block 的 cell 数
    # orientations: 方向梯度数
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    return features


# 提取训练集 HOG 特征
print("正在提取训练集 HOG 特征...")
train_features = []
for i, img in enumerate(train_images):
    features = extract_hog_features(img)
    train_features.append(features)
    if (i + 1) % 1000 == 0:
        print(f"  已处理 {i + 1}/{len(train_images)} 张图片")

train_features = np.array(train_features)
train_labels = np.array(train_labels)

print(f"训练集特征形状: {train_features.shape}")

# 提取测试集 HOG 特征
print("正在提取测试集 HOG 特征...")
test_features = []
for i, img in enumerate(test_images):
    features = extract_hog_features(img)
    test_features.append(features)
    if (i + 1) % 500 == 0:
        print(f"  已处理 {i + 1}/{len(test_images)} 张图片")

test_features = np.array(test_features)
test_labels = np.array(test_labels)

print(f"测试集特征形状: {test_features.shape}")

# 记录特征提取结束时间和内存
feature_end_time = time.time()
mem_after_feature = process.memory_info().rss / 1024 / 1024  # MB

feature_time = feature_end_time - feature_start_time
feature_mem_usage = mem_after_feature - mem_before_feature

print(f"\n特征提取总耗时: {feature_time:.2f} 秒")
print(f"特征提取内存消耗: {feature_mem_usage:.2f} MB")


# ============================================
# 第三部分：SVM 模型训练
# ============================================
print("\n" + "=" * 60)
print("第三步：训练 SVM 模型")
print("=" * 60)

# 记录训练开始时间和内存
train_start_time = time.time()
mem_before_train = process.memory_info().rss / 1024 / 1024  # MB

# 创建并训练 SVM 分类器
# 使用 RBF 核函数，默认参数
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

print("正在训练 SVM（这可能需要几分钟）...")
svm_model.fit(train_features, train_labels)

# 记录训练结束时间和内存
train_end_time = time.time()
mem_after_train = process.memory_info().rss / 1024 / 1024  # MB

train_time = train_end_time - train_start_time
train_mem_usage = mem_after_train - mem_before_train

print(f"\nSVM 训练总耗时: {train_time:.2f} 秒")
print(f"SVM 训练内存消耗: {train_mem_usage:.2f} MB")


# ============================================
# 第四部分：模型评估
# ============================================
print("\n" + "=" * 60)
print("第四步：模型评估")
print("=" * 60)

# 在测试集上进行预测
print("正在测试集上进行预测...")
predictions = svm_model.predict(test_features)

# 计算准确率
accuracy = accuracy_score(test_labels, predictions)
print(f"\n测试集准确率 (Accuracy): {accuracy:.4f} ({accuracy * 100:.2f}%)")

# 打印详细分类报告
print("\n分类报告 (Classification Report):")
print(classification_report(
    test_labels,
    predictions,
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
print(f"特征提取耗时: {feature_time:.2f} 秒")
print(f"SVM 训练耗时: {train_time:.2f} 秒")
print(f"总耗时: {feature_time + train_time:.2f} 秒")
print(f"\n特征提取内存峰值变化: {feature_mem_usage:.2f} MB")
print(f"SVM 训练内存峰值变化: {train_mem_usage:.2f} MB")
print(f"总内存消耗: {feature_mem_usage + train_mem_usage:.2f} MB")

print("\n" + "=" * 60)
print("程序执行完成！")
print("=" * 60)