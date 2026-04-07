import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. 再次加载数据集（因为数据已经下载，这次它会直接从 D 盘读取）
data_root = 'D:/ML_Data/cifar10'
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                        download=False, transform=transform)

# 取出前一批数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=0)
dataiter = iter(trainloader)
images, labels = next(dataiter)

# 定义 CIFAR-10 的真实标签类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# 2. 编写绘图函数
def imshow(img):
    npimg = img.numpy()
    # PyTorch 的图像张量格式是 (C, H, W)，即 (颜色通道, 高, 宽)
    # 而 matplotlib 绘图需要的格式是 (H, W, C)，所以需要转换一下维度
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 3. 显示图像和对应的标签
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(16)))
imshow(torchvision.utils.make_grid(images))