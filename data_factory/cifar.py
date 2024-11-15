from torchvision import (
    datasets
)

from data_factory.base_loader import DataManager


class CIFARDataset(DataManager):
    def __init__(self, dataset: str, transform=None):
        super().__init__(dataset=dataset, transform=transform)

    def get_dataset(self):
        """
        DownLoads the correct data from internet
        :return: dataset_train, dataset_test
        """
        # ----------------------------------------
        # Image Classification Datasets
        #   1. CIFAR10
        #   2. CIFAR100
        # ----------------------------------------
        if self.data_set == 'cifar10':
            print("Creating Classification Dataset - CIFAR 10")
            dataset_train = datasets.CIFAR10(
                "datasets/cifar10",
                transform=self.transform,
                train=True,
                download=True
            )
            dataset_test = datasets.CIFAR10(
                "datasets/cifar10",
                transform=self.transform,
                train=False,
                download=True
            )
            self.num_classes = 10
            self.mean, self.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
            self.num_channels, self.height, self.width, self.input_shape = 3, 32, 32, 32

        elif self.data_set == 'cifar100':
            print("Creating Classification Dataset - CIFAR 100")
            dataset_train = datasets.CIFAR100(
                "datasets/cifar100",
                transform=self.transform,
                train=True,
                download=True
            )
            dataset_test = datasets.CIFAR100(
                "datasets/cifar100",
                transform=self.transform,
                train=False,
                download=True
            )
            self.num_classes = 100
            self.mean, self.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
            self.num_channels, self.height, self.width, self.input_shape = 3, 32, 32, 32

        else:
            raise NotImplementedError

        return dataset_train, dataset_test