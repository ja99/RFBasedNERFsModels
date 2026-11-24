"""Beta-weighted heteroscedastic Gaussian regression loss (Seitzer et al., ICLR 2022)."""


def beta_gaussian_loss(mean, variance, target, beta):
    """Mean over outputs and examples; the caller supplies strictly positive variances."""
    if not 0 <= beta <= 1:
        raise ValueError("Beta must lie in [0, 1]")
    squared_error = (target - mean).square()
    nll = 0.5 * (variance.log() + squared_error / variance)
    return (variance.detach().pow(beta) * nll).mean()
