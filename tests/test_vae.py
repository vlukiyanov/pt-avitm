import torch

from ptavitm.vae import copy_embeddings_, prior, ProdLDA


def test_prior():
    # check against the code in https://git.io/fhL6y
    prior_mean, prior_var = prior(50)
    assert prior_mean.allclose(prior_mean.new().resize_as_(prior_mean).fill_(0.0))
    assert prior_var.allclose(prior_var.new().resize_as_(prior_var).fill_(0.98))


def test_copy_embeddings():
    def embedding(word):
        return torch.ones(300) if word == 'test' else None

    lookup = {**{index: 'z'*index for index in range(9)}, **{9: 'test'}}
    module = torch.nn.Linear(10, 300)
    with torch.no_grad():
        module.weight.copy_(torch.zeros(300, 10))
        copy_embeddings_(module.weight, embedding, lookup)
    assert module.weight[:, 9].eq(1).all()
    for index in range(9):
        assert module.weight[:, index].eq(0).all()


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
