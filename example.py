from data_factory import DataFactory


# Example usage:
if __name__ == '__main__':
    TEST_DATASETS = [
        # 'cifar10',
        # 'cifar100',
        'flickr8k'
    ]
    from torch.utils.data import DataLoader

    for d in TEST_DATASETS:
        data = DataFactory(d)
        tr_data, te_data = data.get_dataset()

        tr_loader = DataLoader(tr_data, batch_size=32, shuffle=True)
        te_loader = DataLoader(te_data, batch_size=32, shuffle=False)