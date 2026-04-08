"""
SVM 基线模型 - CIFAR-10 图像分类
使用 HOG 特征进行分类
增加样本量 + 多次实验取平均
"""

import time
import numpy as np
import torchvision
import torchvision.transforms as transforms
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from multiprocessing import Pool, cpu_count
import psutil
import os


def extract_hog_features(image):
    """从单张图片中提取 HOG 特征"""
    img = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
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


def extract_features_parallel(images, num_workers=None):
    """使用多进程并行提取 HOG 特征"""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    with Pool(processes=num_workers) as pool:
        features = pool.map(extract_hog_features, images)
    return np.array(features)


def run_svm_experiment():
    TRAIN_SAMPLES = 10000
    TEST_SAMPLES = 1000
    NUM_EXPERIMENTS = 3

    print("=" * 60)
    print("SVM + HOG 实验配置")
    print("=" * 60)
    print(f"训练集样本数: {TRAIN_SAMPLES}")
    print(f"测试集样本数: {TEST_SAMPLES}")
    print(f"重复实验次数: {NUM_EXPERIMENTS}")
    print(f"使用 CPU 核心数: {cpu_count()}")

    print("\n" + "=" * 60)
    print("第一步：加载 CIFAR-10 数据集")
    print("=" * 60)

    transform = transforms.Compose([
        transforms.ToTensor()
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

    all_accuracies = []
    all_feature_times = []
    all_train_times = []
    all_total_times = []
    all_feature_mems = []
    all_train_mems = []

    for exp_id in range(NUM_EXPERIMENTS):
        print("\n" + "=" * 60)
        print(f"第 {exp_id + 1}/{NUM_EXPERIMENTS} 次实验")
        print("=" * 60)

        random_seed = 42 + exp_id * 100
        np.random.seed(random_seed)

        train_indices = np.random.choice(len(train_dataset), TRAIN_SAMPLES, replace=False)
        test_indices = np.random.choice(len(test_dataset), TEST_SAMPLES, replace=False)

        print(f"随机种子: {random_seed}")
        print(f"训练集样本数: {len(train_indices)}")
        print(f"测试集样本数: {len(test_indices)}")

        train_images = [train_dataset[i][0].numpy() for i in train_indices]
        train_labels = [train_dataset[i][1] for i in train_indices]

        test_images = [test_dataset[i][0].numpy() for i in test_indices]
        test_labels = [test_dataset[i][1] for i in test_indices]

        print("\n" + "-" * 40)
        print("第二步：提取 HOG 特征（多进程加速）")
        print("-" * 40)

        process = psutil.Process(os.getpid())

        feature_start_time = time.time()
        mem_before_feature = process.memory_info().rss / 1024 / 1024

        print(f"正在提取训练集 HOG 特征 ({TRAIN_SAMPLES} 张)...")
        train_features = extract_features_parallel(train_images)
        print(f"训练集特征形状: {train_features.shape}")

        print(f"正在提取测试集 HOG 特征 ({TEST_SAMPLES} 张)...")
        test_features = extract_features_parallel(test_images)
        print(f"测试集特征形状: {test_features.shape}")

        feature_end_time = time.time()
        mem_after_feature = process.memory_info().rss / 1024 / 1024

        feature_time = feature_end_time - feature_start_time
        feature_mem_usage = mem_after_feature - mem_before_feature

        print(f"特征提取总耗时: {feature_time:.2f} 秒")
        print(f"特征提取内存消耗: {feature_mem_usage:.2f} MB")

        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        print("\n" + "-" * 40)
        print("第三步：训练 SVM 模型")
        print("-" * 40)

        train_start_time = time.time()
        mem_before_train = process.memory_info().rss / 1024 / 1024

        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

        print("正在训练 SVM...")
        svm_model.fit(train_features, train_labels)

        train_end_time = time.time()
        mem_after_train = process.memory_info().rss / 1024 / 1024

        train_time = train_end_time - train_start_time
        train_mem_usage = mem_after_train - mem_before_train

        print(f"SVM 训练总耗时: {train_time:.2f} 秒")
        print(f"SVM 训练内存消耗: {train_mem_usage:.2f} MB")

        print("\n" + "-" * 40)
        print("第四步：模型评估")
        print("-" * 40)

        print("正在测试集上进行预测...")
        predictions = svm_model.predict(test_features)

        accuracy = accuracy_score(test_labels, predictions)
        print(f"\n测试集准确率 (Accuracy): {accuracy:.4f} ({accuracy * 100:.2f}%)")

        if exp_id == NUM_EXPERIMENTS - 1:
            print("\n分类报告 (Classification Report):")
            print(classification_report(
                test_labels,
                predictions,
                target_names=[
                    'airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck'
                ]
            ))

        all_accuracies.append(accuracy * 100)
        all_feature_times.append(feature_time)
        all_train_times.append(train_time)
        all_total_times.append(feature_time + train_time)
        all_feature_mems.append(feature_mem_usage)
        all_train_mems.append(train_mem_usage)

        print(f"\n本次实验总计: 耗时 {feature_time + train_time:.2f} 秒, 内存 {feature_mem_usage + train_mem_usage:.2f} MB")

    print("\n" + "=" * 60)
    print("统计结果汇总")
    print("=" * 60)

    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_feature_time = np.mean(all_feature_times)
    mean_train_time = np.mean(all_train_times)
    mean_total_time = np.mean(all_total_times)
    mean_feature_mem = np.mean(all_feature_mems)
    mean_train_mem = np.mean(all_train_mems)
    mean_total_mem = np.mean(all_feature_mems) + np.mean(all_train_mems)

    print(f"\n准确率 (Accuracy):")
    print(f"  各次实验: {[f'{a:.2f}%' for a in all_accuracies]}")
    print(f"  平均值 ± 标准差: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")

    print(f"\n特征提取耗时 (秒):")
    print(f"  各次实验: {[f'{t:.2f}' for t in all_feature_times]}")
    print(f"  平均值: {mean_feature_time:.2f} 秒")

    print(f"\nSVM 训练耗时 (秒):")
    print(f"  各次实验: {[f'{t:.2f}' for t in all_train_times]}")
    print(f"  平均值: {mean_train_time:.2f} 秒")

    print(f"\n总耗时 (秒):")
    print(f"  各次实验: {[f'{t:.2f}' for t in all_total_times]}")
    print(f"  平均值: {mean_total_time:.2f} 秒")

    print(f"\n内存消耗 (MB):")
    print(f"  特征提取平均: {mean_feature_mem:.2f} MB")
    print(f"  训练平均: {mean_train_mem:.2f} MB")
    print(f"  总计平均: {mean_total_mem:.2f} MB")

    print("\n" + "=" * 60)
    print("程序执行完成！")
    print("=" * 60)


if __name__ == '__main__':
    run_svm_experiment()