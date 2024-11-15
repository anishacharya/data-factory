from data_factory.flickr import FlickrDataset
from data_factory.cifar import CIFARDataset

class DataFactory:
    data_map = {
        'cifar10': CIFARDataset,
        'cifar100': CIFARDataset,
        'flickr8k': FlickrDataset,
    }

    def __init__(self, dataset: str, transform=None):
        self.dataset = dataset
        self.transform = transform

    def get_data(self):
        return self.data_map[self.dataset](self.dataset, self.transform).get_dataset()


if __name__ == '__main__':
    data = DataFactory('cifar10')
    tr_data, te_data = data.get_data()