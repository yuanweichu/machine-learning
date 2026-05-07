import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

data_root = 'D:/ML_Data/cifar10'
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                        download=False, transform=transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=0)
dataiter = iter(trainloader)
images, labels = next(dataiter)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

print(' '.join(f'{classes[labels[j]]:5s}' for j in range(16)))
imshow(torchvision.utils.make_grid(images))