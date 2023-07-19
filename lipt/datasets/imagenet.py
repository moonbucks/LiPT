from torchvision import datasets, transforms
from torch import tensor, long
import os


def ImageNet(data_path):
    channel = 3
    im_size = (224, 224)
    num_classes = 1000
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    #dst_train = datasets.ImageNet(data_path, split="train", transform=transforms.Compose([
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    dst_train = datasets.ImageFolder(data_path, traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
#            transforms.Resize(256),
#            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    #dst_test = datasets.ImageNet(data_path, split="val", transform=transforms.Compose([
    dst_test = datasets.ImageFolder(data_path, valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    class_names = dst_train.classes
    #dst_train.targets = tensor(dst_train.targets, dtype=long)
    #dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
