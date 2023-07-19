from torchvision import datasets, transforms
from torch import tensor, long


def CIFAR10(data_path, custom_transforms=None):
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    if custom_transforms:
        transform = custom_transforms
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), 
      transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=train_transform)
    dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=val_transform)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
