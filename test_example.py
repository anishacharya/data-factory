import pytest
from data_factory import DataFactory
from torch.utils.data import DataLoader

# Test valid dataset names
@pytest.mark.parametrize("dataset_name", ["cifar10", "cifar100", "flickr8k"])
def test_datafactory_initialization(dataset_name):
    try:
        data = DataFactory(dataset_name)
        tr_data, te_data = data.get_dataset()
        assert tr_data is not None, "Training dataset is None"
        assert te_data is not None, "Test dataset is None"
    except Exception as e:
        pytest.fail(f"DataFactory initialization failed for {dataset_name}: {e}")

# Test invalid dataset names
@pytest.mark.parametrize("dataset_name", ["unknown_dataset", "", None])
def test_datafactory_invalid_dataset(dataset_name):
    with pytest.raises(ValueError):
        DataFactory(dataset_name)

# Test DataLoader functionality
@pytest.mark.parametrize("batch_size", [1, 32, 128])
def test_dataloader_functionality(batch_size):
    data = DataFactory("flickr8k")
    tr_data, te_data = data.get_dataset()
    tr_loader = DataLoader(tr_data, batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(te_data, batch_size=batch_size, shuffle=False)
    
    # Check batch size and DataLoader length
    for batch in tr_loader:
        assert len(batch[0]) <= batch_size, "Training batch size mismatch"
        break  # Only checking first batch for simplicity

    for batch in te_loader:
        assert len(batch[0]) <= batch_size, "Test batch size mismatch"
        break

# Test edge case: empty dataset (mock scenario)
def test_empty_dataset_handling(monkeypatch):
    def mock_get_dataset():
        return [], []
    monkeypatch.setattr(DataFactory, "get_dataset", mock_get_dataset)

    data = DataFactory("flickr8k")
    tr_data, te_data = data.get_dataset()
    assert len(tr_data) == 0, "Training dataset should be empty"
    assert len(te_data) == 0, "Test dataset should be empty"
