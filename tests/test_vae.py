import torch

from ptavitm.vae import prior, ProdLDA


def test_prior():
    # check against the code in https://git.io/fhL6y
    prior_mean, prior_var = prior(50)
    assert prior_mean.allclose(prior_mean.new().resize_as_(prior_mean).fill_(0.0))
    assert prior_var.allclose(prior_var.new().resize_as_(prior_var).fill_(0.98))


def test_forward_dimensions():
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


def test_parameters():
    vae = ProdLDA(
        in_dimension=10,
        hidden1_dimension=20,
        hidden2_dimension=10,
        topics=5
    )
    # encoder
    # two each for the linear units
    assert len(tuple(vae.encoder.parameters())) == 4
    assert len(tuple(param for param in vae.encoder.parameters() if param.requires_grad)) == 4
    # mean and logvar
    # two for the linear, two for the batchnorm
    assert len(tuple(vae.mean.parameters())) == 4
    assert len(tuple(param for param in vae.mean.parameters() if param.requires_grad)) == 3
    assert len(tuple(vae.logvar.parameters())) == 4
    assert len(tuple(param for param in vae.logvar.parameters() if param.requires_grad)) == 3
    # decoder
    # one for the linear, two for the batchnorm
    assert len(tuple(vae.decoder.parameters())) == 3
    # batchnorm has no scale
    assert len(tuple(param for param in vae.decoder.parameters() if param.requires_grad)) == 2
