from data_factory.base_loader import DataManager
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


class Flickr(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        if transform:
            self.transform = transform
        else:
            # Since Flickr images are of different sizes, we resize them to 224x224
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        img = example["jpg"]
        target_captions = example["txt"]

        # Convert the image to a PIL Image
        img = Image.open(img)

        # Apply the transformations if specified
        if self.transform:
            img = self.transform(img)

        return img, target_captions


class FlickrDataset(DataManager):
    def __init__(self, dataset: str, transform=None):
        super().__init__(dataset=dataset, transform=transform)

    def get_dataset(self):
        if self.data_set == 'flickr8k':
            dataset = load_dataset("clip-benchmark/wds_flickr8k")
            train_dataset = Flickr(dataset["train"], transform=self.transform)
            test_dataset = Flickr(dataset["test"], transform=self.transform)
            return train_dataset, test_dataset
        else:
            raise NotImplementedError("Dataset not implemented")