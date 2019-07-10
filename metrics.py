from typing import Callable

import torch
import torch.distributions as dist

from losses import kl_divergence_normal_and_spherical


def reparametrize(mu, logvar, logits):
    std = (0.5 * logvar).exp()  # shape: batch_size x n_mixtures x z_dim
    # get rid of 1st dimension since n_mixtures = 1
    return dist.normal.Normal(loc=mu.squeeze(1),
                                scale=std.squeeze(1)).rsample()


def log_likelihood(x, encoder: Callable, decoder: Callable, n_samples: int = 100):
    """
    Importance weighted estimate of marginal log density.

    log p(x) ~  log { 1/K sum_{i=1}^K [exp{ log p(x,z_i) - log q(z_i|x)}] }

        where z_i ~ q(z|x) for i = 1 to 100
    """
    batch_size = x.size(0)
    z_mu, z_logvar, logits = encoder(x)

    log_w = []
    for i in range(n_samples):
        z_i = reparametrize(z_mu, z_logvar, logits)
        x_mu_i = decoder(z_i)
        log_p_x_given_z_i = bernoulli_log_pdf(x.view(batch_size, -1),
                                                x_mu_i.view(batch_size, -1))
        kl_div_i = kl_divergence_normal_and_spherical(z_mu, z_logvar)

        log_w_i = log_p_x_given_z_i - kl_div_i
        log_w_i = log_w_i.unsqueeze(1)
        log_w_i = log_w_i.cpu()
        log_w.append(log_w_i)
    log_w = torch.cat(log_w, dim=1)
    log_w = log_mean_exp(log_w, dim=1)

    return log_w