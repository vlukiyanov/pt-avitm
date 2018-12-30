from ptavitm.model import predict, train
import torch
from torch.utils.data import TensorDataset
from unittest.mock import Mock


def test_train():
    autoencoder = Mock()
    autoencoder.return_value = [torch.tensor([1, 1], dtype=torch.float)] * 3
    optimizer = Mock()
    dataset = TensorDataset(torch.zeros(100, 1000))
    train(
        dataset,
        autoencoder,
        1,
        10,
        optimizer
    )
    autoencoder.train.assert_called_once()
    assert autoencoder.call_count == 10
    assert optimizer.zero_grad.call_count == 10
    assert optimizer.step.call_count == 10


def test_predict():
    # only tests the encode=True
    autoencoder = Mock()
    autoencoder.encode.return_value = torch.tensor([1], dtype=torch.float)
    dataset = TensorDataset(torch.zeros(100, 1000))
    output = predict(dataset, autoencoder, batch_size=10)
    assert autoencoder.encode.call_count == 10
    assert output.shape == (10,)
