"""The gradient noise scale's two terms, free of torch so the analysis readers can import them."""


def noise_scale_terms(g2_small, g2_big, b_small, b_big):
    """McCandlish et al. 2018 (arXiv:1812.06162) eq. A.2: unbiased (tr Sigma, |G|^2) from g2_small, the mean squared
    norm of gradients over batches of b_small examples, and g2_big, the squared norm of a gradient over b_big examples
    (E|G_B|^2 = |G|^2 + tr Sigma / B). Either can come out negative on one measurement: average S and |G|^2 over
    measurements separately, then B_simple = S / |G|^2."""
    g2 = (b_big * g2_big - b_small * g2_small) / (b_big - b_small)
    s = (g2_small - g2_big) / (1.0 / b_small - 1.0 / b_big)
    return s, g2
