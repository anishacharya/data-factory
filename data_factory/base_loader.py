from torch.utils.data import Dataset


class DataManager:
    """
    Data Manager Class
    """

    def __init__(
            self,
            dataset: str,
            transform=None
    ):
        """
        Data Manager
        :param dataset: str, name of the dataset
        """
        # --- data config ---
        self.data_set = dataset
        self.transform = transform

        # --- initialize attributes specific to dataset
        self.num_classes = None
        self.num_channels, self.height, self.width, self.input_shape = None, None, None, None
        self.mean, self.std = None, None

    # ---- main function to get datasets -----
    def get_dataset(self) -> [Dataset] * 2:
        """
        DownLoads the correct data from internet

        :return: dataset_train, dataset_test
        """
        raise NotImplementedError("Subclass must implement abstract method")