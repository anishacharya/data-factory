from data_factory.flickr import FlickrDataset
from data_factory.cifar import CIFARDataset

class DataFactory:
    data_map = {
        'cifar10': CIFARDataset,
        'cifar100': CIFARDataset,
        'flickr8k': FlickrDataset,
    }

    def __init__(self, dataset: str, transform=None, cache_dir: str=None):
        """
        Initializes the DataFactory to act as the specified dataset class.

        :param dataset: str, the name of the dataset (e.g., 'cifar10').
        :param transform: Callable or None, the transformation to be applied.
        """
        if dataset not in self.data_map:
            raise ValueError(
                f"Dataset {dataset} is not supported. Supported datasets: "
                f"{list(self.data_map.keys())}"
            )

        self._dataset_instance = self.data_map[dataset](
            dataset=dataset,
            transform=transform,
            cache_dir=cache_dir
        )

    def __getattr__(self, attr):
        """
        Delegate attribute access to the dataset instance.

        :param attr: str, the attribute to retrieve.
        :return: The requested attribute or method from the dataset instance.
        """
        return getattr(self._dataset_instance, attr)




