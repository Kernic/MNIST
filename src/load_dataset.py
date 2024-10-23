from torchvision.transforms import v2, ElasticTransform, RandomRotation, ToTensor, RandomResizedCrop
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def get_datasets(batchsize=64, transform=False):
    compose = (
        v2.Compose([
            RandomRotation(7),
            RandomResizedCrop(28, scale=(0.89, 1.0), ratio=(0.89, 1.11)),
            ElasticTransform(alpha=37.0, sigma=5.5),
            #v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
            ToTensor(),
        ])
        if transform
        else v2.Compose([ToTensor()])
    )
    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=compose
    )
    return DataLoader(
        train_dataset, 
        batch_size=batchsize
    ), DataLoader(
        test_dataset, 
        batch_size=batchsize
    )
