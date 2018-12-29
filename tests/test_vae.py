import pytest
import torch

from ptavitm.vae import prior, ProdLDA


def test_prior():
    # TODO close enough but needs further testing
    prior_mean, prior_var = prior(50)
    #print(prior_mean)
    #print(prior_var)


def test_dimensions():
    vae = ProdLDA(
        in_dimension=10,
        hidden1_dimension=20,
        hidden2_dimension=10,
        topics=5
    )
    for size in [10, 100, 1000]:
        batch = torch.zeros(size, 10)
        recon, mean, logvar = vae(batch)
        assert recon.shape == batch.shape
        assert mean.shape == (size, 5)
        assert logvar.shape == (size, 5)
