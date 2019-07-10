import torch


def bernoulli_log_pdf(x: torch.Tensor, mu: torch.Tensor):
    """
    Log probability distribution function for Bernoulli distributions.

        Let pi be logits.
        pdf     = pi^x * (1 - pi)^(1-x)
        log_pdf = x * log(pi) + (1 - x) * log(1 - pi)

    In practice, we need to clamp pi (the logits) because if it
    becomes 0 or 1, then log(0) will be nan.
    """
    mu = torch.clamp(mu, 1e-7, 1.-1e-7)
    return torch.sum(x * torch.log(mu) + (1. - x) * torch.log(1. - mu), dim=1)


def kl_divergence_normal_and_spherical(mu, logvar):
    """
    Closed-form representation of KL divergence between a N(mu, sigma)
    posterior and a spherical Gaussian N(0, 1) prior.

    See https://arxiv.org/abs/1312.6114 for derivation.
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def elbo(x, x_mu, z, z_mu, z_logvar, logits):
    """
    Evidence lower bound objective on marginal log density.

    log p(x) > E_{q(z|x)}[log p(x,z) - log q(z|x)]
             = E_{q(z|x)}[log p(x|z)] - KL(q(z|x)||p(z))
    """
    batch_size = x.size(0)
    log_p_x_given_z = bernoulli_log_pdf(x.view(batch_size, -1),
                                        x_mu.view(batch_size, -1))
    kl_div = kl_divergence_normal_and_spherical(z_mu, z_logvar)
    kl_div = torch.sum(kl_div, dim=1)
    elbo = log_p_x_given_z - kl_div
    elbo = torch.mean(elbo)

    # important to negate so that we have a positive loss
    return -elbo