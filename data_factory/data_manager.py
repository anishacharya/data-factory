from torch.utils.data import Dataset
from torchvision import datasets, transforms


class DataManager:
    """
    Data Manager Class
    """

    def __init__(
            self,
            dataset: str,
    ):
        """
        Data Manager

        :param dataset: str, name of the dataset
        """
        # --- data config ---
        self.data_set = dataset

        # --- initialize attributes specific to dataset
        self.num_classes = None
        self.num_channels, self.height, self.width, self.input_shape = None, None, None, None
        self.mean, self.std = None, None

    # ---- main function to get datasets -----
    def get_dataset(self, aug: transforms = None) -> [Dataset] * 2:
        """
        DownLoads the correct data from internet

        :param aug: torch transformation to be applied to the dataset
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
                transform=aug,
                train=True,
                download=True
            )
            dataset_test = datasets.CIFAR10(
                "datasets/cifar10",
                transform=aug,
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
                transform=aug,
                train=True,
                download=True
            )
            dataset_test = datasets.CIFAR100(
                "datasets/cifar100",
                transform=aug,
                train=False,
                download=True
            )
            self.num_classes = 100
            self.mean, self.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
            self.num_channels, self.height, self.width, self.input_shape = 3, 32, 32, 32

        else:
            raise NotImplementedError

        return dataset_train, dataset_test