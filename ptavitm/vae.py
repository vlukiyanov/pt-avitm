from collections import OrderedDict

import torch
import torch.nn as nn

from typing import Callable, Mapping, Optional, Tuple


def prior(topics: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prior for the model.

    :param topics: number of topics
    :return: mean and variance tensors
    """
    a = torch.Tensor(1, topics).float().fill_(1.0)
    mean = a.log().t() - a.log().mean(1)
    var = ((1 - 2.0 / topics) * a.reciprocal()).t() + (1.0 / topics ** 2) * a.reciprocal().sum(1)
    return mean.t(), var.t()


def encoder(in_dimension: int,
            hidden1_dimension: int,
            hidden2_dimension: int,
            encoder_noise: float = 0.2) -> torch.nn.Module:
    return nn.Sequential(OrderedDict([
        ('linear1', nn.Linear(in_dimension, hidden1_dimension)),
        ('act1', nn.Softplus()),
        ('linear2', nn.Linear(hidden1_dimension, hidden2_dimension)),
        ('act2', nn.Softplus()),
        ('dropout', nn.Dropout(encoder_noise))
    ]))


def copy_embeddings_(tensor: torch.Tensor,
                     embedding: Callable[[str], Optional[torch.Tensor]],
                     lookup: Mapping[int, str]) -> None:
    """
    Helper function for mutating the weight of an initial linear embedding module using
    precomputed word vectors.

    :param tensor: weight Tensor to mutate of shape [embedding_dimension, features]
    :param embedding: callable which given a word string returns its embedding Tensor or None if out of vocabulary
    :param lookup: given an index return the corresponding word string
    :return: None
    """
    for index in range(tensor.shape[1]):
        current_embedding = embedding(lookup[index])
        if current_embedding is not None:
            tensor[:, index].copy_(current_embedding)


def decoder(in_dimension: int,
            topics: int,
            decoder_noise: float,
            eps: float,
            momentum: float) -> torch.nn.Module:
    return nn.Sequential(OrderedDict([
        ('linear', nn.Linear(topics, in_dimension, bias=False)),
        ('batchnorm', nn.BatchNorm1d(in_dimension, affine=True, eps=eps, momentum=momentum)),
        ('act', nn.Softmax(dim=1)),
        ('dropout', nn.Dropout(decoder_noise))
    ]))


def hidden(hidden2_dimension: int,
           topics: int,
           eps: float,
           momentum: float) -> torch.nn.Module:
    return nn.Sequential(OrderedDict([
        ('linear', nn.Linear(hidden2_dimension, topics)),
        ('batchnorm', nn.BatchNorm1d(topics, affine=True, eps=eps, momentum=momentum))
    ]))


class ProdLDA(nn.Module):
    def __init__(self,
                 in_dimension: int,
                 hidden1_dimension: int,
                 hidden2_dimension: int,
                 topics: int,
                 decoder_noise: float = 0.2,
                 encoder_noise: float = 0.2,
                 batchnorm_eps: float = 0.001,
                 batchnorm_momentum: float = 0.001) -> None:
        super(ProdLDA, self).__init__()
        self.topics = topics
        self.encoder = encoder(in_dimension, hidden1_dimension, hidden2_dimension, encoder_noise)
        self.mean = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.logvar = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.decoder = decoder(
            in_dimension, topics, decoder_noise=decoder_noise, eps=batchnorm_eps, momentum=batchnorm_momentum
        )
        # set the priors, do not learn them
        self.prior_mean, self.prior_var = prior(topics)
        self.prior_logvar = self.prior_var.log()
        self.prior_mean.requires_grad = False
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False
        # do not learn the batchnorm weight, setting it to 1 as in https://git.io/fhtsY
        for component in [self.mean, self.logvar, self.decoder]:
            component.batchnorm.weight.requires_grad = False
            component.batchnorm.weight.fill_(1.0)
        # initialize decoder weight
        nn.init.xavier_uniform_(self.decoder.linear.weight, gain=1)

    def encode(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(batch)
        return encoded, self.mean(encoded), self.logvar(encoded)

    def decode(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Same as _generator_network method in the original TensorFlow implementation at https://git.io/fhUJu
        eps = mean.new().resize_as_(mean).normal_(mean=0, std=1)
        z = mean + logvar.exp().sqrt() * eps
        return self.decoder(z)

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, mean, logvar = self.encode(batch)
        recon = self.decode(mean, logvar)
        return recon, mean, logvar

    def loss(self,
             input_tensor: torch.Tensor,
             reconstructed_tensor: torch.Tensor,
             posterior_mean: torch.Tensor,
             posterior_logvar: torch.Tensor) -> torch.Tensor:
        """
        Variational objective, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017,
        https://arxiv.org/pdf/1703.01488.pdf; modified from https://github.com/hyqneuron/pytorch-avitm.

        :param input_tensor: input batch to the network, shape [batch size, features]
        :param reconstructed_tensor: reconstructed batch, shape [batch size, features]
        :param posterior_mean: posterior mean
        :param posterior_logvar: posterior log variance
        :return: unaveraged loss tensor
        """
        # TODO check this again against TF implementation adding tests
        # https://github.com/akashgit/autoencoding_vi_for_topic_models/blob/master/models/prodlda.py
        # reconstruction loss
        rl = -(input_tensor * (reconstructed_tensor + 1e-10).log()).sum(1)
        # KL divergence
        prior_mean = self.prior_mean.expand_as(posterior_mean)
        prior_var = self.prior_var.expand_as(posterior_logvar)
        prior_logvar = self.prior_logvar.expand_as(posterior_logvar)
        var_division = posterior_logvar.exp() / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        kld = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.topics)
        return rl + kld
