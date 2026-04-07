import torch
import torchvision
import torchvision.transforms as transforms
import os

def prepare_data():
    # 1. 路径设置：确保存在 D 盘，避免占用 C 盘
    data_root = 'D:/ML_Data/cifar10'
    if not os.path.exists(data_root):
        os.makedirs(data_root)
        print(f"创建目录: {data_root}")

    # 2. 定义数据转换（标准化）
    # CIFAR-10 图片是 32x32 像素的彩色图
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("正在检查/下载数据集，请稍候...")

    # 3. 下载并加载训练集
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                            download=True, transform=transform)
    
    # 4. 下载并加载测试集
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                           download=True, transform=transform)

    # 5. 定义类别名称（用于后续论文描述）
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')

    print("-" * 30)
    print(f"数据存放在: {data_root}")
    print(f"训练集样本数: {len(trainset)}")
    print(f"测试集样本数: {len(testset)}")
    print(f"数据集类别: {classes}")
    print("-" * 30)
    print("运行成功！你可以开始编写模型代码了。")

if __name__ == "__main__":
    prepare_data()