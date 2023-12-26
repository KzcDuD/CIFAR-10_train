import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

batch_size =6
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

img_dataset = torchvision.datasets.CIFAR10(root='./data',train=False,
                                             download=True, transform=transform)

img_loader = torch.utils.data.DataLoader(img_dataset , batch_size=batch_size , shuffle=True)

def imshow(img):
    img =img /2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter =iter(img_loader)
images , labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))