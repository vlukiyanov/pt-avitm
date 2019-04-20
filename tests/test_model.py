from ptavitm.model import perplexity, predict, train
import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock


def test_train():
    autoencoder = Mock()
    autoencoder.return_value = [torch.tensor([1, 1], dtype=torch.float)] * 3
    autoencoder.loss.return_value = torch.tensor([1, 1], dtype=torch.float).requires_grad_()
    optimizer = Mock()
    dataset = TensorDataset(torch.zeros(100, 1000))
    train(
        dataset=dataset,
        autoencoder=autoencoder,
        epochs=1,
        batch_size=10,
        optimizer=optimizer
    )
    autoencoder.train.assert_called_once()
    assert autoencoder.call_count == 10
    assert optimizer.zero_grad.call_count == 10
    assert optimizer.step.call_count == 10


def test_train_validation():
    autoencoder = Mock()
    autoencoder.return_value = [torch.zeros(10, 10).float()] * 3
    autoencoder.loss.return_value = torch.zeros(10, 10).float().requires_grad_()
    optimizer = Mock()
    dataset = TensorDataset(torch.zeros(100, 1000))
    validation_dataset = TensorDataset(torch.zeros(10, 1000))
    train(
        dataset=dataset,
        validation=validation_dataset,
        autoencoder=autoencoder,
        epochs=1,
        batch_size=10,
        optimizer=optimizer
    )
    assert autoencoder.train.call_count == 2
    assert autoencoder.call_count == 11
    assert optimizer.zero_grad.call_count == 10
    assert optimizer.step.call_count == 10


def test_perplexity():
    dataset = TensorDataset(torch.ones(10, 1000))
    validation_loader = DataLoader(
        dataset,
        batch_size=10,
        pin_memory=False,
        sampler=None,
        shuffle=False
    )
    autoencoder = Mock()
    autoencoder.return_value = [torch.ones(10, 10).float()] * 3
    autoencoder.loss.return_value = torch.ones(10, 10).float().requires_grad_()
    assert isinstance(perplexity(validation_loader, autoencoder), float)


def test_predict():
    # only tests the encode=True
    autoencoder = Mock()
    autoencoder.forward.return_value = torch.tensor([1], dtype=torch.float)
    dataset = TensorDataset(torch.zeros(100, 1000))
    output = predict(dataset, autoencoder, batch_size=10, encode=False)
    assert autoencoder.forward.call_count == 10
    assert output.shape == (10,)
